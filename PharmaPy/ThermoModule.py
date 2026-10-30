# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:28:24 2020

@author: dcasasor
"""

import json
import pathlib
import warnings
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from PharmaPy.Errors import PharmaPyValueError


_DatabasePath = Union[str, pathlib.Path]

# Absolute closure tolerance for solvent-blend mass fractions z_j (sum_j z_j == 1).
_SOLVENT_FRACTION_SUM_ATOL = 1e-6  # [-], tolerance for solvent-blend mass-fraction closure

# Phase and thermophysical fields whose last axis is indexed by self.name_species.
_SPECIES_INDEXED_PHASE_FIELDS = frozenset({
    'mass_frac', 'mole_frac', 'mole_conc', 'mass_conc',
    'mw', 'rho_liq', 'rho_solid', 't_crit',
    'delta_hvap', 'tref_hvap', 'henry_constant', 'visc_gas',
})


class FieldLookupError(KeyError, ValueError):
    """Requested phase field or species name does not exist on the phase."""

    def __str__(self) -> str:
        """Return the unquoted message string for both KeyError and ValueError."""
        return str(self.args[0]) if self.args else ''


VALID_ACTIVITY_MODELS = ('ideal', 'UNIFAC', 'UNIQUAC')

# Database key of the pure-component heat-capacity polynomial of each phase.
_CP_PROPERTY_BY_PHASE = {'liquid': 'cp_liq', 'solid': 'cp_solid',
                         'vapor': 'cp_vapor'}

# Database key of the pure-component density [kg/m**3] of each phase that the
# density methods accept. Vapor densities come from the ideal-gas law
# (VaporPhase.getDensity), not from a stored property, so 'vapor' is rejected.
_DENSITY_PROPERTY_BY_PHASE = {'liquid': 'rho_liq', 'solid': 'rho_solid'}

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

# Meaning and unit of the scalar (one value per species) properties whose
# absence the property methods report, quoted in missing-data errors.
_SCALAR_PROPERTY_DESCRIPTIONS = {
    'rho_liq': 'pure liquid density',
    'rho_solid': 'pure solid density',
    't_crit': 'critical temperature',
    'henry_constant': 'Henry constant',
    'surf_tension': 'surface tension',
    'visc_gas': 'vapor viscosity',
    'ri': 'UNIQUAC volume parameter',
    'qi': 'UNIQUAC surface-area parameter',
    'qip': 'UNIQUAC residual surface-area parameter',
}
_SCALAR_PROPERTY_UNITS = {
    'rho_liq': '[kg/m**3]',
    'rho_solid': '[kg/m**3]',
    't_crit': '[K]',
    'henry_constant': '([Pa], liquid mole-fraction basis)',
    'surf_tension': '[N/m]',
    'visc_gas': '[Pa*s]',
    'ri': '[-]',
    'qi': '[-]',
    'qip': '[-]',
}

# Species-record keys of a property database that PharmaPy reads, with the
# unit each one is read in: mw [g/mol]; rho_liq, rho_solid [kg/m**3];
# cp_liq, cp_solid, cp_vapor (ascending polynomial [J/mol/K]); visc_liq
# (log10 mu [mPa*s] correlation); visc_gas [Pa*s]; p_vap (Antoine, [Pa]);
# diffusivity [m**2/s]; t_crit [K]; henry_constant [Pa]; surf_tension
# [N/m]; delta_hvap [J/mol] at tref_hvap [K] (also given as
# {"value": ..., "temp_ref": ...}); mol_vol [m**3/mol]; ri, qi, qip
# (UNIQUAC, [-]). p_crit [Pa] is the documented critical-pressure key of the
# database template (doc/online_docs/examples/PharmaPy_guide.rst); it is
# recognized so that its misspellings are reported, although no current
# calculation reads it. Any other key (Formula, CAS, source, units, ...) is
# metadata: ParseDatabase stores it and no calculation reads it.
RECOGNIZED_PROPERTY_KEYS = frozenset({
    'mw', 'rho_liq', 'rho_solid', 'cp_liq', 'cp_solid', 'cp_vapor',
    'visc_liq', 'visc_gas', 'p_vap', 'diffusivity', 't_crit', 'p_crit',
    'henry_constant', 'surf_tension', 'delta_hvap', 'tref_hvap', 'mol_vol',
    'ri', 'qi', 'qip',
})

# Near-miss spellings of recognized keys that shipped databases use, mapped to
# the recognized key (workshop_data/Case_study_1 and Case_study_2). A key that
# differs from a recognized key only in letter case or in the separators
# '_', '-' and ' ' (for example 'MW' or 'T_crit') is also a near miss; see
# _near_miss_key. Explicit rules replace a fuzzy similarity threshold, so
# a metadata key is never flagged by accident.
_PROPERTY_KEY_ALIASES = {
    'cp_vap': 'cp_vapor',
    'Tc': 't_crit',
    'temp_crit': 't_crit',
    'Pc': 'p_crit',
    'surface_tension': 'surf_tension',
}


class MissingPropertyError(AttributeError, ValueError):
    """Property data that a result needs are missing or unusable.

    This is the one exception of the property methods for missing or invalid
    property data. It is raised when a species that a result needs has no
    data for a property (a NaN entry or row, see ``ParseDatabase``, or a
    key that no species supplies) or has data that the correlation cannot
    use (for example a Watson reference with ``tref_hvap >= t_crit``). The
    message names the species and the property. It covers list-valued
    coefficients (``cp_*``, ``visc_liq``, ``p_vap``, ``diffusivity``),
    scalar properties (``rho_liq``, ``rho_solid``, ``t_crit``,
    ``henry_constant``, ``surf_tension``, ``visc_gas``), the Watson
    latent-heat data (``delta_hvap``, ``tref_hvap``), and a list-valued
    property given as one scalar per species instead of a coefficient list
    (rejected whenever it is evaluated, whatever the fractions). Other scalar
    properties keep their earlier behavior; for example ``getMolVolMix``
    still returns NaN for a missing ``mol_vol`` (issue #511).

    It subclasses ``AttributeError``, the type raised for a missing property
    before #414, and ``ValueError``, the type raised for missing
    ``visc_gas`` (#89) and invalid Watson data (#178) before #424, so
    existing ``except`` clauses and ``pytest.raises`` checks for either type
    still catch it. A malformed database file is reported when it is parsed
    instead (``ParseDatabase`` raises ``TypeError`` for a property given as
    a scalar by one species and as a list by another).

    Reactor solves (``Reactors._BaseReactor._simulate``) use the subclass to
    recognize it and re-raise it, chained from the CVode error, when the
    solver hides the original exception. In other solver-based unit
    operations such an error can still be lost behind the solver's error,
    or stall the solver, which issue #435 tracks.
    """


class PropertyKeyWarning(UserWarning):
    """A property-database key looks like a misspelled recognized key.

    Emitted by ``ParseDatabase`` for a species-record key that is not in
    ``RECOGNIZED_PROPERTY_KEYS`` but is a near miss of one of them (see
    ``_near_miss_key``). PharmaPy does not read the near-miss key, so a
    calculation that needs the property reports it as missing. Filter this
    category to silence the warning for a database known to be correct.
    """


def _near_miss_key(key: str) -> Optional[str]:
    """Return the recognized property key that ``key`` nearly spells.

    Parameters
    ----------
    key : str
        Species-record key of a property database.

    Returns
    -------
    str or None
        The recognized key that ``key`` is a near miss of, or ``None`` when
        ``key`` is recognized itself or is not a near miss (metadata).

    Notes
    -----
    A key is a near miss if ``_PROPERTY_KEY_ALIASES`` lists it, or if it
    equals a recognized key after both are lowercased and stripped of
    every character other than letters and digits. The rules are exact, so
    the result is deterministic and needs no similarity threshold.
    """
    if key in RECOGNIZED_PROPERTY_KEYS:
        return None

    if key in _PROPERTY_KEY_ALIASES:
        return _PROPERTY_KEY_ALIASES[key]

    folded = _folded_key(key)
    for recognized in sorted(RECOGNIZED_PROPERTY_KEYS):
        if _folded_key(recognized) == folded:
            return recognized

    return None


def _folded_key(name: str) -> str:
    """Fold a database key for the case- and separator-blind comparison.

    Parameters
    ----------
    name : str
        Database key.

    Returns
    -------
    str
        ``name`` lowercased, keeping only its letters and digits; for
        example ``'T_crit'`` and ``'t-crit'`` both give ``'tcrit'``.
    """
    return ''.join(char for char in name.lower() if char.isalnum())


def _warn_near_miss_keys(records: Dict[str, Any],
                         path_datafile: Any) -> None:
    """Warn about near-miss keys in the species records of a database.

    Parameters
    ----------
    records : dict
        Species name mapped to its property record, as loaded from JSON,
        without the ``interaction`` entry.
    path_datafile : path-like or sequence of path-like
        Database path or paths, quoted in the warning.

    Warns
    -----
    PropertyKeyWarning
        One warning per near-miss key, naming the key, the species that use
        it and the recognized key, in order of first appearance.
    """
    species_by_key = {}
    for name, record in records.items():
        for key in record:
            if _near_miss_key(key) is not None:
                species_by_key.setdefault(key, []).append(name)

    for key, names in species_by_key.items():
        warnings.warn(
            f"Property key {key!r} of species {names} in {path_datafile!r} "
            f"is not read by PharmaPy and looks like the recognized key "
            f"{_near_miss_key(key)!r}; its value is ignored, so a calculation "
            "that needs the property reports it as missing. Rename the key "
            "if it holds that property (see "
            "PharmaPy.ThermoModule.RECOGNIZED_PROPERTY_KEYS).",
            PropertyKeyWarning, stacklevel=3)


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

    Warns
    -----
    PropertyKeyWarning
        For each species-record key that is a near miss of a recognized
        property key, such as ``cp_vap`` for ``cp_vapor`` or ``Tc`` for
        ``t_crit``; the message names the key, the species and the
        recognized key. Metadata keys (``Formula``, ``CAS``, ``source``,
        ``units``, ...) do not warn.

    Notes
    -----
    The recognized species-record keys, with their units, are
    ``RECOGNIZED_PROPERTY_KEYS``. Every key is stored, recognized or not,
    but only recognized keys are read by the property methods. A near miss
    is a key listed in ``_PROPERTY_KEY_ALIASES`` (the misspellings found in
    shipped databases) or one that matches a recognized key apart from
    letter case and the separators ``_``, ``-`` and space; its value is
    never used in place of the recognized key.

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
    ``AttributeError`` and ``ValueError``) naming the species and the
    property when a result needs a NaN row, or a NaN entry of one of the
    scalar properties that ``MissingPropertyError`` lists. Other scalar
    properties keep their earlier behavior, for example a NaN
    ``getMolVolMix`` for a missing ``mol_vol`` (issue #511).

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
    every species that supplies the property must supply a list. When every
    species that supplies a property gives a scalar (or ``{"value":
    scalar}``), the property keeps the one-dimensional representation, a
    float array of shape ``(num_species,)`` with NaN for species that omit
    it, and loads without error or warning. For the recognized list-valued
    properties (``cp_liq``, ``cp_solid``, ``cp_vapor``, ``visc_liq``,
    ``p_vap`` and ``diffusivity``) that representation is malformed: a
    scalar is not read as a constant correlation, and every evaluation of
    the property raises ``MissingPropertyError`` naming the species, the
    values and the required coefficient list (issue #424, maintainer
    decision "reject when used"). With ``to_arrays=False``, the raw values,
    including ``None`` and empty lists, are returned.
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

    _warn_near_miss_keys(original_data, path_datafile)

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

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            list-valued property (a key of ``_LIST_PROPERTY_DESCRIPTIONS``)
            is stored with fewer than two dimensions, that is, as one scalar
            per species instead of a list of coefficients (see
            ``ParseDatabase``). This is checked before any species selection
            or weighting, so it applies whatever the fractions.

        Notes
        -----
        The dimension is checked before ``np.atleast_2d``, which would turn
        ``N`` scalars into one row of ``N`` coefficients and so hide the
        error whenever ``N`` matches a valid coefficient count (one species,
        three species for Antoine, four for ``visc_liq``).
        """
        if not hasattr(self, prop_name):
            width = _CORRELATION_WIDTHS.get(prop_name, 1)  # [-]
            return np.full((len(self.name_species), width), np.nan)

        stored = getattr(self, prop_name)
        if prop_name in _LIST_PROPERTY_DESCRIPTIONS and np.ndim(stored) < 2:
            self._reject_scalar_list_property(prop_name, stored)

        return np.atleast_2d(stored)

    def _reject_scalar_list_property(self, prop_name: str,
                                     stored: Any) -> None:
        """Report a list-valued property stored as scalars.

        Parameters
        ----------
        prop_name : str
            Database key of the property, a key of
            ``_LIST_PROPERTY_DESCRIPTIONS``.
        stored : float or array-like
            Stored values, one entry per species, shape ``()`` or
            ``(num_species,)``: numbers in the property's units as supplied
            (NaN or ``None`` where a species omits the property), or
            non-numeric placeholders such as strings, which ``ParseDatabase``
            keeps as a list.

        Raises
        ------
        MissingPropertyError
            Always. The message names the property, the species that supply
            scalars and their values (numbers or placeholders, in species
            order), the database, and the coefficient convention the
            property requires. If no species supplies a value (every entry
            is NaN), the property is reported as missing for every species,
            as for a property that the database lacks.
        """
        entries = np.atleast_1d(np.asarray(stored, dtype=object))
        names = []
        shown = []  # property units as supplied (unverified), or placeholders
        for name, entry in zip(self.name_species, entries):
            if entry is None:
                continue
            if isinstance(entry, (int, float, np.number)):
                if np.isnan(entry):
                    continue
                entry = float(entry)
            names.append(name)
            shown.append(entry)

        if not names:
            num_sp = len(self.name_species)
            self._check_species_rows(
                prop_name, np.full(num_sp, np.nan), np.arange(num_sp),
                np.ones(num_sp, dtype=bool),
                'give no value for it, so any calculation that evaluates it')

        raise MissingPropertyError(
            f"Property '{prop_name}' "
            f"({_LIST_PROPERTY_DESCRIPTIONS[prop_name]}) of species {names} "
            f"in the property database {self.path_data!r} is given as the "
            f"scalar value(s) {shown}, but it must be a "
            "list of coefficients per species "
            f"{_LIST_PROPERTY_UNITS[prop_name]}. A scalar is not read as a "
            "constant correlation. Confirm the units and physical basis of "
            "these values and supply the coefficient list that the "
            "correlation requires for each species")

    def _scalar_values(self, prop_name: str) -> np.ndarray:
        """Return a scalar property as one value per species.

        Parameters
        ----------
        prop_name : str
            Database key of the property, for example ``'rho_liq'``.

        Returns
        -------
        ndarray
            The stored values as floats, with the property's units, shape
            ``(num_species,)``; NaN marks a species without data (see
            ``ParseDatabase``). A property that every species omits gives
            NaN for every species, so it takes the same missing-data path.
            The stored array is returned without copying when it is already
            a float array; callers must not modify it.
        """
        if not hasattr(self, prop_name):
            return np.full(len(self.name_species), np.nan)

        return np.asarray(getattr(self, prop_name), dtype=float)

    def _require_property(self, prop_name: str, reason: str) -> np.ndarray:
        """Return a scalar property that the database must supply.

        Parameters
        ----------
        prop_name : str
            Database key of the property, a key of
            ``_SCALAR_PROPERTY_DESCRIPTIONS``.
        reason : str
            Clause completing "Species [...] <reason> needs <property>".

        Returns
        -------
        ndarray
            The stored values as floats, with the property's units, shape
            ``(num_species,)``. NaN entries are returned as they are; the
            caller applies its own rule to them.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If no species supplies the property, naming every species.
        """
        if not hasattr(self, prop_name):
            self._check_species_rows(
                prop_name, np.full(len(self.name_species), np.nan),
                np.arange(len(self.name_species)),
                np.ones(len(self.name_species), dtype=bool), reason)

        return np.asarray(getattr(self, prop_name), dtype=float)

    def _check_species_rows(self, prop_name: str, rows: np.ndarray,
                            species_idx: Sequence[int], needed: np.ndarray,
                            reason: str, remedy: str = '') -> np.ndarray:
        """Flag missing property data and reject the data a result needs.

        Parameters
        ----------
        prop_name : str
            Database key of the property, a key of
            ``_LIST_PROPERTY_DESCRIPTIONS`` or
            ``_SCALAR_PROPERTY_DESCRIPTIONS``.
        rows : ndarray
            Property data of the selected species, in ``species_idx`` order,
            with the units of the property: coefficient rows of shape
            ``(num_selected, num_coefficients)``, or one value per species,
            shape ``(num_selected,)``, for a scalar property.
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
            True where a selected species has no data (its value is NaN or
            its row contains NaN), shape ``(num_selected,)``. Callers give
            such species exactly zero weight when they are not needed,
            instead of letting NaN times a zero fraction turn the result into
            NaN.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If a needed species has no data. The message names the species
            and the property, as for a property that the database lacks
            entirely.
        """
        missing = np.isnan(rows)  # [-]
        if missing.ndim > 1:
            missing = missing.any(axis=1)  # [-], a NaN coefficient marks the row
        lacking = np.flatnonzero(missing & needed)
        if lacking.size > 0:
            names = [self.name_species[species_idx[ind]] for ind in lacking]
            if prop_name in _LIST_PROPERTY_DESCRIPTIONS:
                description = _LIST_PROPERTY_DESCRIPTIONS[prop_name]
                units = _LIST_PROPERTY_UNITS[prop_name]
            else:
                description = _SCALAR_PROPERTY_DESCRIPTIONS[prop_name]
                units = _SCALAR_PROPERTY_UNITS[prop_name]
            raise MissingPropertyError(
                f"Species {names} {reason} needs {description} "
                f"'{prop_name}' {units}, which the property database "
                f"{self.path_data!r} does not supply for them; add "
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

    def _validate_weights_shape(self, weights: np.ndarray,
                                fractions: np.ndarray,
                                fraction_name: str) -> np.ndarray:
        """Require presence weights with exactly the shape of the fractions.

        Parameters
        ----------
        weights : array-like
            Weights [any basis, e.g. 0/1] that decide which species need
            data in place of ``fractions``.
        fractions : array-like
            Fractions [-] that the weights stand in for, shape
            ``(num_species,)`` or ``(num_rows, num_species)``.
        fraction_name : str
            Name of ``fractions`` in the caller's signature, for the message.

        Returns
        -------
        ndarray
            ``weights`` as an array.

        Raises
        ------
        ValueError
            If ``weights`` is not one- or two-dimensional, does not have one
            entry per species on its last axis, or does not have the shape
            of ``fractions``. A scalar or a short array would otherwise
            broadcast and silently switch the missing-data checks off.
        """
        weights = np.asarray(weights)
        weight_shape = weights.shape
        num_species = len(self.name_species)  # [-]
        if (len(weight_shape) not in (1, 2)
                or weight_shape[-1] != num_species
                or weight_shape != np.shape(fractions)):
            raise ValueError(
                f"weights must have the shape of {fraction_name}, "
                f"{np.shape(fractions)}, one entry per species "
                f"({num_species} species) and, for a "
                f"composition profile, one row per row of {fraction_name}; "
                f"got shape {weight_shape}")

        return weights

    def _validate_species_weights(self, weights: np.ndarray) -> np.ndarray:
        """Require one presence weight per species for a pure-component kernel.

        Parameters
        ----------
        weights : array-like
            Weights [any basis, e.g. 0/1] that decide which species a
            pure-component kernel evaluates, one per species in database
            order.

        Returns
        -------
        ndarray
            ``weights`` as an array, shape ``(num_species,)``.

        Raises
        ------
        ValueError
            If ``weights`` does not have exactly the shape
            ``(num_species,)``. A scalar or a short array would otherwise
            broadcast and silently switch the missing-data checks off.
        """
        weights = np.asarray(weights)
        num_species = len(self.name_species)  # [-]
        if weights.shape != (num_species,):
            raise ValueError(
                "weights must have one entry per species, shape "
                f"({num_species},); got shape {weights.shape}")

        return weights

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
            A subclass of ``AttributeError`` and ``ValueError``.
            If a needed species has no coefficients for the phase. Without
            ``weights`` every species is returned, so every species is
            needed.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).

        Notes
        -----
        With ``weights``, a species without coefficients whose weight is zero
        in every row is returned as exactly 0, so that the caller's weighted
        sum is exact (never NaN times zero). That 0 is a placeholder for a
        zero-weight term, not a heat capacity. Nonzero is bitwise: a
        round-off-level weight counts as present.
        """
        prop_name, cp_cts = self._cp_rows(phase)  # [J/mol/K/K**k], ascending k
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
                 basis: str = 'mass',
                 weights: Optional[np.ndarray] = None
                 ) -> Union[float, np.ndarray]:
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
        weights : ndarray, optional
            Presence weights [any basis, e.g. 0/1], with exactly the shape
            of the fractions on the requested basis, ``(num_species,)`` or
            ``(num_temperatures, num_species)``. When given, they replace
            the fractions in deciding which species need coefficients: a
            species with zero weight in every row needs none and, without
            them, contributes exactly zero. The fractions still weight the
            sum. Solver-based units pass a structural presence mask here,
            so that round-off fractions of absent species need no data.

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
            Also raised if ``weights`` does not have exactly the shape of
            the fractions on the requested basis (one- or two-dimensional,
            one entry per species on the last axis), as a scalar or short
            array would otherwise switch the check off.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If a species with a nonzero fraction (or, with ``weights``, a
            nonzero weight) in some composition row has no coefficients for
            the phase. A species whose fraction (weight) is zero in every
            row needs none and contributes exactly zero. Nonzero is
            bitwise: a round-off-level fraction counts as present unless
            ``weights`` says otherwise.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        prop_name, cp_cts = self._cp_rows(phase)  # [J/mol/K/K**k], ascending k
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

        # Only species present in some composition row (or with a nonzero
        # weight) need coefficients; absent species without them contribute
        # exactly zero, not NaN * 0.
        if weights is None:
            needed = np.atleast_2d(np.asarray(frac) != 0).any(axis=0)
            reason = 'carry a nonzero fraction, so the mixture heat capacity'
        else:
            weights = self._validate_weights_shape(
                weights, frac, 'mass_frac' if basis == 'mass' else 'mole_frac')
            needed = self._needed_by_weights(weights, len(cp_cts))
            reason = 'carry a nonzero weight, so the mixture heat capacity'
        missing = self._check_species_rows(
            prop_name, cp_cts, np.arange(len(cp_cts)), needed, reason)
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
            profile. Used only when ``total_h`` is ``True``, which requires
            at least one of them. The fraction matching ``basis`` is used
            when supplied; otherwise the other one is converted.
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
            Amounts or presence flags by which the caller weights the
            species enthalpies [any basis, e.g. mol/L, or 0/1], shape
            ``(num_species,)`` or ``(num_rows, num_species)`` in database
            order. When given, only selected species with a nonzero weight
            in some row need coefficients, and the others are returned (or,
            with ``total_h=True``, weighted) as exactly 0 if they have none.
            With ``total_h=True`` they must have exactly the shape of the
            weighting fractions and replace them in this rule only; the
            fractions still weight the sum. Solver-based units pass
            a structural presence mask here, so that round-off fractions of
            absent species need no data.

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
            ``num_rows`` differ and neither is one. Also raised, before any
            computation, for a ``phase`` other than ``'liquid'``,
            ``'solid'`` or ``'vapor'``, and for ``total_h=True`` when
            neither ``mass_frac`` nor ``mole_frac`` is given or when
            ``weights`` does not have exactly the shape of the weighting
            fractions (one- or two-dimensional, one entry per database
            species on the last axis).
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If a selected species that the result needs has no
            coefficients for the phase (a NaN row, see ``ParseDatabase``).
            With ``total_h=False`` every selected species is needed, or,
            with ``weights``, every one with a nonzero weight. With
            ``total_h=True`` only species with a nonzero fraction in some
            composition row are, or, with ``weights``, a nonzero weight; the
            others contribute exactly zero. Nonzero is bitwise: a
            round-off-level fraction or weight counts as present, including
            round-off that a solver leaves in a fraction that should be
            zero; reactors and evaporators therefore pass a structural
            presence mask as ``weights``.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).

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

        prop_name, cp_cts = self._cp_rows(phase)  # [J/mol/K/K**k], ascending k
        cp_cts = cp_cts[idx]  # [J/mol/K/K**k], ascending k
        species_idx = np.asarray(idx)

        if total_h and mass_frac is None and mole_frac is None:
            raise ValueError(
                "getEnthalpy(total_h=True) weights the species enthalpies by "
                "composition and needs mass_frac or mole_frac [-]; neither "
                "was given (pass total_h=False for species enthalpies)")

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
        for val in temp:
            temp_term = val**exp - temp_ref**exp
            integral.append(np.dot(cp_cts / exp, temp_term))

        integral = np.vstack(integral)

        if total_h:
            if basis == 'mass':
                integralMass = integral * 1000 / self.mw[idx]  # J/kg_i
                if mass_frac is None:
                    mass_frac = self.frac_to_frac(mole_frac=mole_frac)

                if mass_frac.ndim == 1:
                    frac_selected = mass_frac[idx]  # [-]
                else:
                    frac_selected = mass_frac[:, idx]  # [-]

                h_selected = integralMass  # [J/kg]
            elif basis == 'mole':
                if mole_frac is None:
                    mole_frac = self.frac_to_frac(mass_frac)

                if mole_frac.ndim == 1:
                    frac_selected = mole_frac[idx]  # [-]
                else:
                    frac_selected = mole_frac[:, idx]  # [-]

                h_selected = integral  # [J/mol]

            # Only species present in some composition row (or with a
            # nonzero weight) need coefficients; absent species without them
            # contribute exactly zero.
            if weights is None:
                needed = np.atleast_2d(frac_selected != 0).any(axis=0)
                reason = 'carry a nonzero fraction, so the mixture enthalpy'
            else:
                frac_name = 'mass_frac' if basis == 'mass' else 'mole_frac'
                weights = self._validate_weights_shape(
                    weights, mass_frac if basis == 'mass' else mole_frac,
                    frac_name)
                needed = self._needed_by_weights(
                    weights[..., species_idx], len(cp_cts))
                reason = 'carry a nonzero weight, so the mixture enthalpy'
            missing = self._check_species_rows(
                prop_name, cp_cts, species_idx, needed, reason)
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
            A subclass of ``AttributeError`` and ``ValueError``.
            If a selected species with a nonzero stoichiometric coefficient
            in some reaction has no ``cp_liq`` coefficients. A selected
            species that no reaction involves needs none and contributes
            exactly zero.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).
        """
        temp = np.atleast_1d(temp)
        temp_ref = tref_hrxn

        n_temp = len(temp)
        n_rxns, n_species = stoich_matrix.shape

        cp_cts = self._property_rows('cp_liq')[mask]  # [J/mol/K/K**k], ascending k
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

    @staticmethod
    def _validate_density_phase(phase: str) -> str:
        """Check the phase of a density request.

        Parameters
        ----------
        phase : {'liquid', 'solid'}
            Phase whose pure-component density is requested.

        Returns
        -------
        str
            Database key of the phase's pure density [kg/m**3],
            ``'rho_liq'`` or ``'rho_solid'``.

        Raises
        ------
        ValueError
            If ``phase`` is not a key of ``_DENSITY_PROPERTY_BY_PHASE``; this
            includes ``'vapor'``, whose density is not a stored property.
        """
        if phase not in _DENSITY_PROPERTY_BY_PHASE:
            raise ValueError(
                f"phase must be one of {tuple(_DENSITY_PROPERTY_BY_PHASE)}, "
                f"got {phase!r}")

        return _DENSITY_PROPERTY_BY_PHASE[phase]

    def getDensityPure(self, phase: str = 'liquid',
                       temp: Optional[float] = None,
                       weights: Optional[np.ndarray] = None) -> tuple:
        """Return pure-component densities on the mass and molar bases.

        Parameters
        ----------
        phase : {'liquid', 'solid'}, optional
            Selects the ``rho_liq`` or ``rho_solid`` database entry; default
            liquid. Vapor densities are not stored properties (see
            ``VaporPhase.getDensity``), so ``'vapor'`` is rejected.
        temp : float, optional
            Temperature [K]. Accepted for interface compatibility and
            ignored: the stored pure densities are temperature independent.
        weights : ndarray, optional
            Presence weights [any basis, e.g. 0/1], exactly of shape
            ``(num_species,)``. When given, only the species with a nonzero
            weight are evaluated: the others need no density and get
            exactly 0.0 in both returned arrays, a placeholder rather than a
            density, whatever their data, so that nothing computed from the
            weighted species depends on their data. Unit operations pass a
            structural presence mask here (issue #425).

        Returns
        -------
        rho_mass : ndarray
            Pure densities [kg/m**3], shape ``(num_species,)``, in database
            order.
        rho_mole : ndarray
            Pure molar densities [kmol/m**3] (equivalently [mol/L]),
            ``rho_mass / mw`` with ``mw`` in [g/mol], same shape.

        Raises
        ------
        ValueError
            If ``phase`` is not one of ``'liquid'`` or ``'solid'``, checked
            before any computation, or if ``weights`` does not have exactly
            the shape ``(num_species,)``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If any
            species (with ``weights``, any species with a nonzero weight)
            has no density for the phase; without ``weights`` every species
            is returned, so every species needs it.
        """
        prop_name = self._validate_density_phase(phase)
        rhoMass = self._scalar_values(prop_name)  # [kg/m**3]
        num_sp = len(rhoMass)
        if weights is None:
            self._check_species_rows(
                prop_name, rhoMass, np.arange(num_sp),
                np.ones(num_sp, dtype=bool),
                'are evaluated as pure components, so their density')
        else:
            evaluated = self._validate_species_weights(weights) != 0  # [-]
            self._check_species_rows(
                prop_name, rhoMass, np.arange(num_sp), evaluated,
                'carry a nonzero weight, so their pure density')
            # Unweighted species are not evaluated: a 0.0 placeholder.
            rhoMass = np.where(evaluated, rhoMass, 0.0)  # [kg/m**3]

        rhoMole = rhoMass / self.mw  # [kmol/m**3], equivalently [mol/L]

        return rhoMass, rhoMole

    def _inverse_densities(self, phase: str, needed: np.ndarray,
                           reason: str) -> tuple:
        """Return pure specific volumes, zero for unneeded missing data.

        Parameters
        ----------
        phase : {'liquid', 'solid'}
            Phase whose pure densities are inverted.
        needed : ndarray of bool
            True for a species whose volume the result uses, shape
            ``(num_species,)``.
        reason : str
            Clause completing "Species [...] <reason> needs <property>".

        Returns
        -------
        inv_mass : ndarray
            ``1 / rho_i`` [m**3/kg], shape ``(num_species,)``.
        inv_mole : ndarray
            ``1 / (rho_i / mw_i)`` [m**3/kmol] (equivalently [L/mol]), same
            shape.

        Raises
        ------
        ValueError
            If ``phase`` is not one of ``'liquid'`` or ``'solid'``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If a needed species has no density for the phase.

        Notes
        -----
        A species without data that is not needed gets exactly zero in both
        arrays, so a weighted sum in which its weight is zero stays exact
        instead of becoming NaN times zero. That zero is a placeholder, not a
        specific volume.
        """
        prop_name = self._validate_density_phase(phase)
        rho_mass = self._scalar_values(prop_name)  # [kg/m**3]
        missing = self._check_species_rows(
            prop_name, rho_mass, np.arange(len(rho_mass)), needed, reason)
        rho_mole = rho_mass / self.mw  # [kmol/m**3]

        inv_mass = np.where(missing, 0.0, 1 / rho_mass)  # [m**3/kg]
        inv_mole = np.where(missing, 0.0, 1 / rho_mole)  # [m**3/kmol]

        return inv_mass, inv_mole

    def getDensityMix(self, mass_frac: Optional[np.ndarray] = None,
                      mole_frac: Optional[np.ndarray] = None,
                      phase: str = 'liquid', temp: Optional[float] = None,
                      basis: str = 'mass',
                      weights: Optional[np.ndarray] = None
                      ) -> Union[float, np.ndarray]:
        """Mix pure-component densities assuming ideal (additive) volumes.

        Parameters
        ----------
        mass_frac, mole_frac : ndarray, optional
            Species fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. The fraction matching ``basis`` is
            used when supplied; otherwise the other one is converted.
        phase : {'liquid', 'solid'}, optional
            Selects the ``rho_liq`` or ``rho_solid`` pure density
            [kg/m**3]; default liquid. ``'vapor'`` is rejected, as in
            ``getDensityPure``.
        temp : float, optional
            Temperature [K]. Accepted for interface compatibility and
            ignored: the stored pure densities are temperature independent,
            so neither ``temp`` nor a ``temp`` attribute is required.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned density; default mass.
        weights : ndarray, optional
            Presence weights [any basis, e.g. 0/1], with exactly the shape
            of the fractions, ``(num_species,)`` or ``(num_points,
            num_species)``. When given, they replace the
            fractions in deciding which species need a density: a species
            with zero weight in every row needs none and, without data,
            adds exactly zero volume. The fractions still weight the
            volumes. Solver-based units pass a structural presence mask
            here, so that round-off fractions of absent species need no
            data.

        Returns
        -------
        float or ndarray
            Mixture density, [kg/m**3] for ``basis='mass'`` and [kmol/m**3]
            (equivalently [mol/L]) for ``basis='mole'``. Scalar for a
            one-dimensional composition, shape ``(num_points,)`` otherwise.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole', or ``phase`` is not
            one of ``'liquid'`` or ``'solid'``. Both checks precede any
            computation. Also raised if ``weights`` does not have exactly
            the shape of the fractions on the requested basis (one- or
            two-dimensional, one entry per species on the last axis), as a
            scalar or short array would otherwise switch the check off.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            with a nonzero fraction (or, with ``weights``, a nonzero
            weight) in some composition row has no density for the phase.
            A species whose fraction (weight) is zero in every row needs
            none and contributes exactly zero volume. Nonzero is bitwise: a
            round-off-level fraction counts as present unless ``weights``
            says otherwise.

        Notes
        -----
        ``1 / rho_mix = sum_i frac_i / rho_i`` on the selected basis, with
        pure molar densities ``rho_i / mw_i`` [kmol/m**3].
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")

        self._validate_density_phase(phase)

        if basis == 'mass':
            if mass_frac is None:
                mass_frac = self.frac_to_frac(mole_frac=mole_frac)

            frac = mass_frac  # [-]
        else:
            if mole_frac is None:
                mole_frac = self.frac_to_frac(mass_frac=mass_frac)

            frac = mole_frac  # [-]

        if weights is None:
            needed = self._needed_by_weights(frac, self.num_species)
            reason = 'carry a nonzero fraction, so the mixture density'
        else:
            weights = self._validate_weights_shape(
                weights, frac, 'mass_frac' if basis == 'mass' else 'mole_frac')
            needed = self._needed_by_weights(weights, self.num_species)
            reason = 'carry a nonzero weight, so the mixture density'
        inv_mass, inv_mole = self._inverse_densities(
            phase, needed, reason)  # [m**3/kg], [m**3/kmol]

        if basis == 'mass':
            rhoMix = 1 / np.dot(mass_frac, inv_mass)  # [kg/m**3]
        else:
            rhoMix = 1 / np.dot(mole_frac, inv_mole)  # [kmol/m**3]

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
            Temperature [K], scalar or shape ``(num_temps,)``, for the liquid
            branch; defaults to the phase temperature. The vapor branch
            ignores it.

        Returns
        -------
        ndarray
            Pure viscosities [Pa*s]. Liquid: shape ``(num_species,)`` for a
            scalar temperature and ``(num_temps, num_species)`` for an array.
            Vapor: shape ``(num_species,)``, the database's ``visc_gas``
            entries in species order.

        Raises
        ------
        ValueError
            If ``phase`` is unknown.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If any species has no liquid ``visc_liq`` coefficients, or, for
            the vapor, no ``visc_gas`` value [Pa*s] (the error names those
            species); every species is returned, so every species needs
            them.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).

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
            visc_gas = self._scalar_values('visc_gas')  # [Pa*s]
            lacking = np.flatnonzero(np.isnan(visc_gas))
            if lacking.size > 0:
                names = [self.name_species[ind] for ind in lacking]
                raise MissingPropertyError(
                    "Vapor viscosity requires the missing 'visc_gas' property "
                    f"[Pa*s] of species {names}. Supply a pure-component "
                    "value for every species in the property JSON "
                    f"{self.path_data!r}, valid at the requested conditions.")
            viscosity = self.visc_gas  # [Pa*s]

        return viscosity

    def _evaluate_liquid_viscosity(self,
                                   temp: Optional[Union[float, np.ndarray]],
                                   visc_cts: np.ndarray) -> np.ndarray:
        """Evaluate the liquid-viscosity correlation without checking data.

        Parameters
        ----------
        temp : float, numpy.ndarray or None
            Temperature [K], scalar or shape ``(num_temps,)``; the phase
            temperature when ``None``.
        visc_cts : ndarray
            Coefficients ``A, B, C, D`` of
            ``log10(mu / [mPa*s]) = A + B/T + C*T + D*T**2``, with units
            [-], [K], [1/K] and [1/K**2]; shape ``(num_species, 4)``.

        Returns
        -------
        ndarray
            Pure liquid viscosities [Pa*s], shape ``(num_species,)`` for a
            scalar temperature and ``(num_temps, num_species)`` otherwise;
            NaN for a species with a NaN row.
        """
        if temp is None:
            temp = self.temp  # [K]

        if np.ndim(temp) == 0:
            temp_term = np.array([1, 1/temp, temp, temp**2])  # [-], [1/K], [K], [K**2]
            log_visc = np.dot(visc_cts, temp_term)  # [-], log10(mu/[mPa*s])
        else:
            temp = np.asarray(temp, dtype=float)  # [K], shape (num_temps,)
            temp_term = np.stack([np.ones_like(temp), 1/temp, temp, temp**2],
                                 axis=-1)  # [-], [1/K], [K], [K**2]
            log_visc = temp_term @ visc_cts.T  # [-], (num_temps, num_species)

        viscosity = 10**log_visc/1000  # [Pa*s], 1000 mPa*s per Pa*s

        return viscosity

    def getViscosityMix(self, temp: Optional[Union[float, np.ndarray]] = None,
                        mass_frac: Optional[np.ndarray] = None,
                        mole_frac: Optional[np.ndarray] = None,
                        phase: str = 'liquid',
                        weights: Optional[np.ndarray] = None
                        ) -> Union[float, np.ndarray]:
        """Mix dynamic viscosities on the mole-fraction basis.

        Parameters
        ----------
        temp : float or ndarray, optional
            Temperature [K], scalar or shape ``(num_temps,)``, for the liquid
            branch; defaults to the phase temperature. The vapor branch
            ignores it.
        mass_frac, mole_frac : ndarray, optional
            Mass or mole fractions [-], shape (num_species,) or
            (num_points, num_species). Mole fractions take precedence;
            if neither is supplied, use the phase mole fractions.
        phase : {'liquid', 'vapor'}, optional
            Select logarithmic liquid mixing or the dilute-gas Wilke rule.
        weights : ndarray, optional
            Presence weights [any basis, e.g. 0/1], with exactly the shape
            of the fractions used (``mole_frac`` when given, else
            ``mass_frac``, else the phase mole fractions). When given, the
            mixing rule is evaluated on the species with a nonzero weight
            only, as if the others were absent: an entry with zero weight
            contributes nothing, whatever its fraction, and a species whose
            weight is zero in every row needs no viscosity data (for the
            vapor, no ``visc_gas``, which the Wilke sums then do not use).
            Solver-based units pass a structural presence mask here, so
            that round-off fractions of absent species need no data (issue
            #425).

        Returns
        -------
        float or ndarray
            Mixture viscosity [Pa*s]. With a scalar temperature: a scalar
            for one composition or shape (num_points,) for a composition
            profile. With a temperature array, the liquid viscosities of each
            temperature are mixed with the composition as in
            ``getEnthalpy``: shape ``(num_temps,)`` for one composition, and
            row-paired when ``num_temps == num_points``.

        Raises
        ------
        ValueError
            If ``phase`` is unknown, or if ``weights`` does not have exactly
            the shape of the fractions used.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            For the vapor, if any species (with ``weights``, any species
            with a nonzero weight) has no ``visc_gas`` value [Pa*s], as in
            ``getViscosityPure``. For the liquid, if a species with a
            nonzero mole fraction (with ``weights``, a nonzero weight) in
            some composition row has no ``visc_liq`` coefficients. A
            species whose fraction (weight) is zero in every row needs none
            and contributes exactly zero to the logarithmic mixing rule.
            Nonzero is bitwise: a round-off-level fraction counts as present
            unless ``weights`` says otherwise.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).

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
        if phase not in ('liquid', 'vapor'):
            raise ValueError(
                f"phase must be one of ('liquid', 'vapor'), got {phase!r}")

        mw_species = self.mw  # [g/mol]
        if phase == 'liquid':
            visc_cts = self._property_rows('visc_liq')  # [-], [K], [1/K], [1/K**2]
            visc_comp = self._evaluate_liquid_viscosity(temp, visc_cts)  # [Pa*s]
        elif weights is None:
            visc_comp = self.getViscosityPure(phase, temp)  # [Pa*s]

        fraction_name = 'mole_frac'
        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac
        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)
            fraction_name = 'mass_frac'

        reason = 'carry a nonzero fraction, so the mixture viscosity'
        if weights is None:
            needed = self._needed_by_weights(mole_frac, len(self.name_species))
        else:
            weights = self._validate_weights_shape(weights, mole_frac,
                                                   fraction_name)
            needed = self._needed_by_weights(weights, len(self.name_species))
            reason = 'carry a nonzero weight, so the mixture viscosity'
            # Zero-weight entries are evaluated as absent.
            mole_frac = np.where(weights != 0, mole_frac, 0.0)  # [-]
            if phase == 'vapor':
                visc_gas = self._scalar_values('visc_gas')  # [Pa*s]
                self._check_species_rows(
                    'visc_gas', visc_gas, np.arange(len(visc_gas)), needed,
                    reason)
                # The Wilke sums run over the weighted species only.
                evaluated = np.flatnonzero(needed)
                visc_comp = visc_gas[evaluated]  # [Pa*s]
                mw_species = np.asarray(self.mw)[evaluated]  # [g/mol]
                mole_frac = mole_frac[..., evaluated]  # [-]

        # Mixing rules
        if phase == 'liquid':
            # Absent species without coefficients contribute exactly zero.
            missing = self._check_species_rows(
                'visc_liq', visc_cts, np.arange(len(visc_cts)), needed,
                reason)
            log_visc = np.where(missing, 0.0, np.log(visc_comp))  # [-], ln(mu/[Pa*s])
            if log_visc.ndim == 1:
                viscMix = np.exp(
                    np.dot(mole_frac, log_visc))
            else:
                # One row per temperature, paired with composition rows.
                viscMix = np.exp((log_visc * mole_frac).sum(axis=-1))  # [Pa*s]

        elif phase == 'vapor':
            if visc_comp.ndim == 1:
                visc_term = np.outer(visc_comp, 1/visc_comp)  # [-]
                mw_term = np.outer(mw_species, 1/mw_species)  # [-], M_i/M_j

                phi_mix = (1 + visc_term**0.5 * (1/mw_term)**0.25)**2 / \
                    np.sqrt(8*(1 + mw_term))  # [-]

                interactions = np.dot(mole_frac, phi_mix.T)  # [-]

                viscMix = (visc_comp * mole_frac / interactions).sum(axis=-1)  # [Pa*s]

        return viscMix

    def getDiffusivityPure(self, wrt: int,
                           temp: Optional[float] = None,
                           weights: Optional[np.ndarray] = None
                           ) -> np.ndarray:
        """Return species diffusivities with respect to one species.

        Parameters
        ----------
        wrt : int
            Index of the reference species, usually the solvent; selects a
            column of the ``diffusivity`` rows.
        temp : float, optional
            Temperature [K]; unused, the stored values are constants.
        weights : ndarray, optional
            Presence weights [any basis, e.g. 0/1], exactly of shape
            ``(num_species,)``. When given, only the species with a nonzero
            weight are evaluated: the others need no diffusivity and get
            exactly 0.0, a placeholder rather than a diffusivity, whatever
            their data. Unit operations pass a structural presence mask
            here (issue #425).

        Returns
        -------
        ndarray
            Diffusivity of every species in ``wrt`` [m**2/s], shape
            ``(num_species,)``.

        Raises
        ------
        ValueError
            If ``weights`` does not have exactly the shape
            ``(num_species,)``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If any species (with ``weights``, any species with a nonzero
            weight) has no ``diffusivity`` data in column ``wrt``; without
            ``weights`` every species is returned, so every species needs
            it.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).
        """
        rows = self._property_rows('diffusivity')  # [m**2/s]
        if hasattr(self, 'diffusivity'):
            rows = rows[:, [wrt]]  # [m**2/s], the values returned below

        num_sp = len(rows)
        if weights is None:
            self._check_species_rows(
                'diffusivity', rows, np.arange(num_sp),
                np.ones(num_sp, dtype=bool),
                'are evaluated as pure components, so their diffusivity')

            diffusivity = self.diffusivity[:, wrt]  # [m**2/s]
            return diffusivity

        evaluated = self._validate_species_weights(weights) != 0  # [-]
        self._check_species_rows(
            'diffusivity', rows, np.arange(num_sp), evaluated,
            'carry a nonzero weight, so their diffusivity')
        # Unweighted species are not evaluated: a 0.0 placeholder.
        return np.where(evaluated, rows[:, 0], 0.0)  # [m**2/s]

    def frac_to_conc(self, mass_frac: Optional[np.ndarray] = None,
                     mole_frac: Optional[np.ndarray] = None,
                     basis: str = 'mole') -> np.ndarray:
        """Convert liquid fractions into ideal-mixing concentrations.

        Parameters
        ----------
        mass_frac, mole_frac : ndarray, optional
            Liquid species fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. ``mass_frac`` takes precedence;
            one of them is required.
        basis : str, optional
            ``'mass'`` returns mass concentrations; any other value,
            including the default ``'mole'``, returns molar concentrations.

        Returns
        -------
        ndarray
            Species concentrations, [mol/L] (equivalently [kmol/m**3]) or
            [kg/m**3] for ``basis='mass'``, with the shape of the given
            fractions.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species with a nonzero fraction in some row has no ``rho_liq``.
            A species whose fraction is zero in every row needs none and
            gets a zero concentration.

        Notes
        -----
        The liquid volume follows ideal mixing with the pure liquid
        densities ``rho_liq``: ``c_i = x_i / sum_j x_j / rho_mole_j`` on the
        molar basis, with ``rho_mole_j = rho_liq_j / mw_j`` [kmol/m**3].
        """
        frac = mole_frac if mass_frac is None else mass_frac  # [-]
        invMass, invMole = self._inverse_densities(
            'liquid', self._needed_by_weights(frac, self.num_species),
            'carry a nonzero fraction, so their concentration')  # [m**3/kg], [m**3/kmol]
        if mass_frac is None:
            if mole_frac.ndim == 1:
                concentr = mole_frac / np.dot(mole_frac, invMole)  # [mol/L]
            else:
                concentr = mole_frac.T / np.dot(mole_frac, invMole)
                concentr = concentr.T  # [mol/L]
        else:
            if mass_frac.ndim == 1:
                concentr = (mass_frac / self.mw) / np.dot(mass_frac,
                                                          invMass)  # [mol/L]
            else:
                concentr = (mass_frac / self.mw).T / np.dot(mass_frac,
                                                            invMass)
                concentr = concentr.T  # [mol/L]

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

    def _coerce_composition_mapping(
        self,
        comp: Any,
        name: str = 'composition',
    ) -> Any:
        """Convert a species-keyed composition mapping into a database-ordered array.

        Parameters
        ----------
        comp : mapping of str to float/array-like, or array-like, or None
            Species composition values in the caller's basis (fractions [-],
            molar concentrations [mol/L], or mass concentrations [kg/m**3]).
            Non-mapping inputs are returned unchanged.
        name : str, optional
            Argument name included in validation error messages.

        Returns
        -------
        numpy.ndarray or Any
            When ``comp`` is a mapping, a float64 array of shape
            ``(num_species,)`` for scalar entries or ``(num_points, num_species)``
            for 1-D profile entries, with omitted species initialized to ``0.0``.
            Otherwise ``comp`` is returned unchanged.

        Raises
        ------
        ValueError
            If ``comp`` is an empty mapping, contains unknown species names, or
            has non-finite, negative, or shape-mismatched values.
        """
        if comp is None or not isinstance(comp, Mapping):
            return comp

        if len(comp) == 0:
            raise ValueError(f"{name} mapping must not be empty.")

        unknown = [
            key for key in comp.keys()
            if not isinstance(key, str) or key not in self.name_species
        ]
        if unknown:
            raise ValueError(
                f"Unknown species {unknown} in {name}; available species are "
                f"{self.name_species}."
            )

        parsed_entries: dict[int, np.ndarray] = {}
        ndims = set()
        profile_lengths = set()
        for sp_name, raw_val in comp.items():
            arr = np.asarray(raw_val, dtype=float)  # [-], [mol/L], or [kg/m**3]
            if arr.ndim > 1 or arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(
                    f"{name} entry for species {sp_name!r} must be a finite "
                    "non-negative scalar or 1-D profile array."
                )
            ndims.add(arr.ndim)
            if arr.ndim == 1:
                profile_lengths.add(len(arr))
            parsed_entries[self.name_species.index(sp_name)] = arr

        if len(ndims) > 1 or len(profile_lengths) > 1:
            raise ValueError(
                f"{name} mapping entries must all be scalars or 1-D arrays of "
                "identical length."
            )

        if 1 in ndims:
            num_points = next(iter(profile_lengths))
            out = np.zeros((num_points, self.num_species), dtype=float)  # [-], [mol/L], or [kg/m**3]
            for sp_idx, arr in parsed_entries.items():
                out[:, sp_idx] = arr
        else:
            out = np.zeros(self.num_species, dtype=float)  # [-], [mol/L], or [kg/m**3]
            for sp_idx, arr in parsed_entries.items():
                out[sp_idx] = float(arr)

        return out

    def _resolve_solvent_blend_mass_conc(
        self,
        mass_conc: np.ndarray,
        solv_mass_frac: Union[Mapping[str, float], np.ndarray],
    ) -> np.ndarray:
        """Close the ideal-mixture volume balance for a single solvent or solvent blend.

        Parameters
        ----------
        mass_conc : numpy.ndarray
            Solute mass concentrations [kg/m**3], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        solv_mass_frac : mapping of str to float or numpy.ndarray
            Solvent-blend mass fractions [-] on a solute-free basis, either as
            a species-keyed mapping ``{solvent_name: z_j}`` or a 1-D array of
            shape ``(num_species,)``.

        Returns
        -------
        numpy.ndarray
            Complete species mass concentrations [kg/m**3], matching the shape
            of ``mass_conc``, with solvent concentrations filled so that
            ``sum_i C_i / rho_i == 1`` and ``C_j / sum_{k in solv} C_k == z_j``.

        Raises
        ------
        IndexError
            If ``mass_conc`` is neither 1-D nor 2-D.
        ValueError
            If solvent fractions are empty, negative, non-finite, do not sum to
            1.0 within ``_SOLVENT_FRACTION_SUM_ATOL``, or overlap with nonzero
            solute concentrations.
        PharmaPyValueError
            If the specified solute concentrations occupy a volume fraction
            ``sum_s C_s / rho_s >= 1.0``, leaving non-positive volume for the
            solvent blend.
        MissingPropertyError
            If a present solute or solvent lacks ``rho_liq`` [kg/m**3].
        """
        if mass_conc.ndim not in (1, 2) or mass_conc.shape[-1] != self.num_species:
            raise IndexError(
                "Solvent-blend back-calculation requires 1-D or 2-D "
                f"concentrations with species axis {self.num_species}, "
                f"got shape {mass_conc.shape}."
            )

        if isinstance(solv_mass_frac, Mapping):
            if len(solv_mass_frac) == 0:
                raise ValueError("Solvent mass_frac mapping must not be empty.")
            unknown = [
                k for k in solv_mass_frac.keys()
                if not isinstance(k, str) or k not in self.name_species
            ]
            if unknown:
                raise ValueError(
                    f"Unknown solvent species {unknown} in mass_frac; "
                    f"available species are {self.name_species}."
                )
            z_solv = np.zeros(self.num_species, dtype=float)  # [-], solvent-blend basis
            solv_mask = np.zeros(self.num_species, dtype=bool)
            for sp_name, raw_val in solv_mass_frac.items():
                val_arr = np.asarray(raw_val, dtype=float)
                if val_arr.ndim != 0 or not np.isfinite(val_arr) or float(val_arr) < 0:
                    raise ValueError(
                        f"Solvent mass_frac for {sp_name!r} must be a finite "
                        "non-negative scalar."
                    )
                sp_idx = self.name_species.index(sp_name)
                z_solv[sp_idx] = float(val_arr)  # [-]
                solv_mask[sp_idx] = True
        else:
            z_solv = np.asarray(solv_mass_frac, dtype=float)  # [-], solvent-blend basis
            if z_solv.shape != (self.num_species,) or not np.all(np.isfinite(z_solv)) or np.any(z_solv < 0):
                raise ValueError(
                    f"solv_mass_frac must be a finite non-negative 1-D array of "
                    f"shape ({self.num_species},)."
                )
            solv_mask = z_solv > 0

        z_sum = float(z_solv.sum())  # [-]
        if abs(z_sum - 1.0) > _SOLVENT_FRACTION_SUM_ATOL:
            raise ValueError(
                f"Solvent-blend mass_frac must sum to 1.0 (got {z_sum:.6g})."
            )
        z_solv = z_solv / z_sum  # [-]
        active_solv_mask = z_solv > 0

        if not np.all(np.isfinite(mass_conc)) or np.any(mass_conc < 0):
            raise ValueError(
                "Solute concentrations must be finite and non-negative."
            )

        any_conc_nonzero = (
            np.any(mass_conc != 0, axis=0)
            if mass_conc.ndim == 2
            else (mass_conc != 0)
        )
        overlap = np.flatnonzero(any_conc_nonzero & solv_mask)
        if overlap.size > 0:
            overlap_names = [self.name_species[idx] for idx in overlap]
            raise ValueError(
                f"Species {overlap_names} cannot be specified with both solute "
                "concentrations and solvent mass_frac."
            )

        needed = any_conc_nonzero | active_solv_mask
        inv_dens, _ = self._inverse_densities(
            'liquid',
            needed,
            'carry a nonzero concentration or are part of the solvent blend, '
            'so the solvent concentration',
        )  # [m**3/kg]

        solute_mask = ~solv_mask
        phi_solutes = np.dot(
            mass_conc[..., solute_mask], inv_dens[solute_mask]
        )  # [-], solute volume fraction
        if np.any(phi_solutes >= 1.0):
            max_phi = float(np.max(phi_solutes))  # [-]
            raise PharmaPyValueError(
                f"Specified solute concentrations occupy volume fraction "
                f"{max_phi:.6g} >= 1.0, leaving non-positive volume for the "
                "solvent blend."
            )

        beta = float(
            np.dot(z_solv[active_solv_mask], inv_dens[active_solv_mask])
        )  # [m**3/kg], pure-solvent-blend specific volume
        total_solv_mass_conc = (1.0 - phi_solutes) / beta  # [kg/m**3]

        filled = mass_conc.copy()  # [kg/m**3]
        if filled.ndim == 1:
            filled[solv_mask] = z_solv[solv_mask] * float(total_solv_mass_conc)  # [kg/m**3]
        else:
            filled[:, solv_mask] = (
                np.asarray(total_solv_mass_conc, dtype=float)[:, None]
                * z_solv[None, solv_mask]
            )  # [kg/m**3]
        return filled

    def retrieve_field(
        self,
        field_name: str,
        species: Optional[Union[str, Sequence[str]]] = None,
        copy: bool = True,
        as_dict: bool = False,
    ) -> Union[float, np.ndarray, dict[str, Union[float, np.ndarray]]]:
        """Retrieve a named composition, property, or state field from the phase.

        Parameters
        ----------
        field_name : str
            Name of the phase field to retrieve, such as ``'mass_frac'`` [-],
            ``'mole_frac'`` [-], ``'mole_conc'`` [mol/L], ``'mass_conc'``
            [kg/m**3], ``'mw'`` [g/mol], ``'temp'`` [K], ``'pres'`` [Pa],
            ``'mass'`` [kg], ``'vol'`` [m**3], ``'moles'`` [mol],
            ``'mass_flow'`` [kg/s], ``'vol_flow'`` [m**3/s], ``'mole_flow'``
            [mol/s], ``'moments'`` [m**n], ``'distrib'`` [#/um], or
            ``'x_distrib'`` [um].
        species : str or sequence of str, optional
            Species name or ordered sequence of species names to extract from a
            species-indexed field. When a single species string is given, a
            scalar float (for 1-D fields) or 1-D spatial profile array is
            returned. When a sequence of species names is given, the last axis
            is reordered/subsetted to match ``species`` in caller order.
        copy : bool, optional
            If True (default), return a copy of array-valued fields so mutating
            the returned array does not corrupt internal phase state. When
            ``copy=False``, whole-field retrieval, single-species retrieval,
            and ``as_dict=True`` profile entries return views into the stored
            array when ``raw_val`` is already a ``float`` ``ndarray``;
            multi-species array subset selection (``as_dict=False`` with a
            sequence of ``species``) always allocates a new array via NumPy
            advanced indexing.
        as_dict : bool, optional
            If True, return species-indexed fields as a dictionary mapping each
            selected species name to its value or spatial profile.

        Returns
        -------
        float, numpy.ndarray, or dict of str to float/numpy.ndarray
            Requested field value in its stored physical units and profile
            axes (1-D ``(num_species,)`` or 2-D ``(num_points, num_species)``
            for species-indexed fields).

        Raises
        ------
        FieldLookupError
            A subclass of ``KeyError`` and ``ValueError``. Raised when
            ``field_name`` does not exist on the phase, is unset, or when
            ``species`` is requested on a non-species field or names an unknown
            species.
        """
        if not isinstance(field_name, str) or not field_name or field_name.startswith('_'):
            raise FieldLookupError(
                f"Field {field_name!r} does not exist on {self.__class__.__name__}."
            )
        if not hasattr(self, field_name):
            raise FieldLookupError(
                f"Field {field_name!r} does not exist on {self.__class__.__name__}."
            )
        raw_val = getattr(self, field_name)
        if raw_val is None or callable(raw_val):
            raise FieldLookupError(
                f"Field {field_name!r} is not available on {self.__class__.__name__}."
            )

        is_species_field = field_name in _SPECIES_INDEXED_PHASE_FIELDS
        if not is_species_field:
            if species is not None:
                raise FieldLookupError(
                    f"Field {field_name!r} on {self.__class__.__name__} is not "
                    "indexed by species."
                )
            if as_dict:
                raise FieldLookupError(
                    f"Field {field_name!r} on {self.__class__.__name__} is not "
                    "indexed by species and cannot be returned as a dictionary."
                )
            if np.ndim(raw_val) == 0:
                return float(raw_val)
            return np.array(raw_val, dtype=float, copy=bool(copy))

        arr = np.array(raw_val, dtype=float, copy=bool(copy))
        if species is None:
            if as_dict:
                return {
                    sp: (float(arr[idx]) if arr.ndim == 1 else arr[..., idx])
                    for idx, sp in enumerate(self.name_species)
                }
            return arr

        if isinstance(species, str):
            if species not in self.name_species:
                raise FieldLookupError(
                    f"Unknown species {species!r}; available species are "
                    f"{self.name_species}."
                )
            sp_idx = self.name_species.index(species)
            selected = arr[..., sp_idx]
            if as_dict:
                return {
                    species: (float(selected) if selected.ndim == 0 else selected)
                }
            if selected.ndim == 0:
                return float(selected)
            return selected.copy() if copy else selected

        if isinstance(species, Sequence):
            unknown = [
                sp for sp in species
                if not isinstance(sp, str) or sp not in self.name_species
            ]
            if unknown:
                raise FieldLookupError(
                    f"Unknown species {unknown}; available species are "
                    f"{self.name_species}."
                )
            indices = [self.name_species.index(sp) for sp in species]
            if as_dict:
                return {
                    sp: (
                        float(arr[idx])
                        if arr.ndim == 1
                        else (arr[..., idx].copy() if copy else arr[..., idx])
                    )
                    for sp, idx in zip(species, indices)
                }
            return arr[..., indices]

        raise FieldLookupError(
            f"species must be a species name string or sequence of strings, "
            f"got {type(species).__name__}."
        )

    def conc_to_frac(self, conc, solvent_ind=None, basis=None,
                     solv_mass_frac=None):
        """Convert liquid molar concentrations into composition fractions.

        Parameters
        ----------
        conc : array-like or mapping of str to float/array-like
            Species molar concentrations [mol/L]. Shape ``(num_species,)``
            is always accepted; ``(num_points, num_species)`` is accepted
            when ``solvent_ind`` is ``None`` (including with
            ``solv_mass_frac``). When ``solvent_ind`` or ``solv_mass_frac``
            is given, the solvent concentration(s) are back-calculated to
            close the mixture volume balance.
        solvent_ind : int or mapping of str to float, optional
            Positional index of the solvent species [-], or a solvent-blend
            mass-fraction mapping ``{solvent_name: z_j}`` [-]. Index ``0`` is a
            valid solvent index, so this argument is compared with ``None``
            and never tested for truth. When ``None`` (and ``solv_mass_frac``
            is ``None``), ``conc`` is taken as a complete composition and no
            solvent concentration is back-calculated.
        basis : {'mole', 'mass', None}, optional
            Basis of the returned fractions. ``'mole'`` returns mole
            fractions only, ``'mass'`` returns mass fractions only, and the
            default ``None`` returns both.
        solv_mass_frac : mapping of str to float or array-like, optional
            Solvent-blend mass fractions [-] on a solute-free basis, summing
            to 1.0. When provided, solvent concentrations are back-calculated
            from the coupled volume-additive equations.

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
            or ``solv_mass_frac`` is not ``None``.

        Raises
        ------
        IndexError
            If ``conc`` is two-dimensional and an integer ``solvent_ind`` is
            not ``None`` (``solv_mass_frac`` supports both 1-D and 2-D
            ``conc`` arrays).
        ValueError
            If a species mapping contains unknown species or invalid solvent
            fractions.
        PharmaPyValueError
            If solute concentrations with ``solv_mass_frac`` exceed the
            available volume (``sum_s c_s * v_s >= 1.0``).
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If
            ``solvent_ind`` or ``solv_mass_frac`` is given and a solvent, or a
            species with a nonzero concentration, has no ``rho_liq``.

        Notes
        -----
        The solvent concentration closes the ideal-mixture volume balance
        ``sum_i c_i * v_i = 1``, in which ``v_i = mw_i / rho_liq_i`` is the
        pure-species molar volume [L/mol] from the ``rho_liq`` data used by
        :meth:`getDensityPure`::

            c_solv = (1 - sum_{i != solv} c_i * v_i) / v_solv

        The length of the returned tuple therefore depends on whether
        ``solvent_ind`` or ``solv_mass_frac`` was supplied, not on its value.

        Warnings
        --------
        ``conc`` is wrapped with ``numpy.asarray`` and the solvent entry is
        written in place, so a NumPy array supplied by the caller is modified
        by this call. A list, tuple, or mapping is copied and left untouched.
        """
        if isinstance(solvent_ind, Mapping) and solv_mass_frac is None:
            solv_mass_frac = solvent_ind
            solvent_ind = None

        if isinstance(conc, Mapping):
            conc = self._coerce_composition_mapping(conc, 'mole_conc')
        else:
            conc = np.asarray(conc)

        has_solvent = (solvent_ind is not None) or (solv_mass_frac is not None)

        if solv_mass_frac is not None:
            mass_conc = np.asarray(conc, dtype=float) * self.mw  # [kg/m**3]
            filled_mass_conc = self._resolve_solvent_blend_mass_conc(
                mass_conc, solv_mass_frac
            )  # [kg/m**3]
            filled_mole_conc = filled_mass_conc / self.mw  # [mol/L]
            if isinstance(conc, np.ndarray) and np.issubdtype(conc.dtype, np.floating):
                conc[:] = filled_mole_conc
            else:
                conc = filled_mole_conc
        elif solvent_ind is not None:
            mask_solv = np.ones_like(conc, dtype=bool)
            mask_solv[solvent_ind] = False

            # The solvent closes the balance, so it always needs a density.
            needed = self._needed_by_weights(conc, self.num_species)  # [-]
            needed[solvent_ind] = True
            _, molVol = self._inverse_densities(
                'liquid', needed,
                'carry a nonzero concentration or are the solvent, so the '
                'solvent concentration')  # [L/mol], equivalently [m**3/kmol]

            conc_solv = (1 - np.dot(conc[mask_solv], molVol[mask_solv])) / \
                molVol[solvent_ind]

            conc[solvent_ind] = conc_solv

        if conc.ndim == 1:
            mole_frac = conc / conc.sum()
        else:
            mole_frac = (conc.T / conc.sum(axis=1)).T

        if basis == 'mole':
            if has_solvent:
                return mole_frac, conc
            else:
                return mole_frac

        elif basis == 'mass':
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
            if has_solvent:
                return mass_frac, conc
            else:
                return mass_frac
        else:
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
            if has_solvent:
                return mass_frac, mole_frac, conc
            else:
                return mass_frac, mole_frac

    def mass_conc_to_frac(self, conc, solvent_ind=None, basis=None,
                          solv_mass_frac=None):
        """Convert liquid mass concentrations into composition fractions.

        Parameters
        ----------
        conc : array-like or mapping of str to float/array-like
            Species mass concentrations [kg/m**3]. Shape ``(num_species,)``
            is always accepted; ``(num_points, num_species)`` is accepted
            when ``solvent_ind`` is ``None`` (including with
            ``solv_mass_frac``). When ``solvent_ind`` or ``solv_mass_frac``
            is given, the solvent concentration(s) are back-calculated to
            close the mixture volume balance.
        solvent_ind : int or mapping of str to float, optional
            Positional index of the solvent species [-], or a solvent-blend
            mass-fraction mapping ``{solvent_name: z_j}`` [-]. Index ``0`` is a
            valid solvent index, so this argument is compared with ``None``
            and never tested for truth. When ``None`` (and ``solv_mass_frac``
            is ``None``), ``conc`` is taken as a complete composition and no
            solvent concentration is back-calculated.
        basis : {'mass', 'mole', None}, optional
            Basis of the returned fractions. ``'mass'`` returns mass
            fractions only, ``'mole'`` returns mole fractions only, and the
            default ``None`` returns both.
        solv_mass_frac : mapping of str to float or array-like, optional
            Solvent-blend mass fractions [-] on a solute-free basis, summing
            to 1.0. When provided, solvent concentrations are back-calculated
            from the coupled volume-additive equations.

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
            ``solvent_ind`` or ``solv_mass_frac`` is not ``None``.

        Raises
        ------
        IndexError
            If ``conc`` is two-dimensional and an integer ``solvent_ind`` is
            not ``None`` (``solv_mass_frac`` supports both 1-D and 2-D
            ``conc`` arrays), for the same reason as in :meth:`conc_to_frac`.
        ValueError
            If a species mapping contains unknown species or invalid solvent
            fractions.
        PharmaPyValueError
            If solute concentrations with ``solv_mass_frac`` exceed the
            available volume (``sum_s C_s / rho_s >= 1.0``).
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If
            ``solvent_ind`` or ``solv_mass_frac`` is given and a solvent, or a
            species with a nonzero concentration, has no ``rho_liq``.

        Notes
        -----
        The solvent concentration closes the ideal-mixture volume balance
        ``sum_i c_i / rho_i = 1``, in which ``rho_i`` is the pure-species mass
        density [kg/m**3] from the ``rho_liq`` data used by
        :meth:`getDensityPure`::

            c_solv = (1 - sum_{i != solv} c_i / rho_i) * rho_solv

        This is the mass-basis counterpart of :meth:`conc_to_frac`, and the
        length of the returned tuple likewise depends on whether
        ``solvent_ind`` or ``solv_mass_frac`` was supplied, not on its value.

        Warnings
        --------
        ``conc`` is wrapped with ``numpy.asarray`` and the solvent entry is
        written in place, so a NumPy array supplied by the caller is modified
        by this call. A list, tuple, or mapping is copied and left untouched.
        """
        if isinstance(solvent_ind, Mapping) and solv_mass_frac is None:
            solv_mass_frac = solvent_ind
            solvent_ind = None

        if isinstance(conc, Mapping):
            conc = self._coerce_composition_mapping(conc, 'mass_conc')
        else:
            conc = np.asarray(conc)

        has_solvent = (solvent_ind is not None) or (solv_mass_frac is not None)

        if solv_mass_frac is not None:
            filled_mass_conc = self._resolve_solvent_blend_mass_conc(
                np.asarray(conc, dtype=float), solv_mass_frac
            )  # [kg/m**3]
            if isinstance(conc, np.ndarray) and np.issubdtype(conc.dtype, np.floating):
                conc[:] = filled_mass_conc
            else:
                conc = filled_mass_conc
        elif solvent_ind is not None:
            mask_solv = np.ones_like(conc, dtype=bool)
            mask_solv[solvent_ind] = False

            # The solvent closes the balance, so it always needs a density.
            needed = self._needed_by_weights(conc, self.num_species)  # [-]
            needed[solvent_ind] = True
            inv_dens, _ = self._inverse_densities(
                'liquid', needed,
                'carry a nonzero concentration or are the solvent, so the '
                'solvent concentration')  # [m**3/kg]
            dens_solvent = self._scalar_values('rho_liq')[solvent_ind]  # [kg/m**3]

            conc_solv = (
                1 - np.dot(conc[mask_solv], inv_dens[mask_solv])) * \
                dens_solvent

            conc[solvent_ind] = conc_solv

        if conc.ndim == 1:
            mass_frac = conc / conc.sum()
        else:
            mass_frac = (conc.T / conc.sum(axis=1)).T

        if basis == 'mass':
            if has_solvent:
                return mass_frac, conc
            else:
                return mass_frac

        elif basis == 'mole':
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)
            if has_solvent:
                return mole_frac, conc
            else:
                return mole_frac

        else:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)
            if has_solvent:
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
            A subclass of ``AttributeError`` and ``ValueError``.
            If a selected species has no ``p_vap`` coefficients; every
            selected species is returned, so every one needs them.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).
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
                      pres: Optional[float] = None,
                      used: Optional[np.ndarray] = None) -> float:
        """Weight Antoine saturation values into a bubble or dew-point seed.

        Parameters
        ----------
        mole_frac : ndarray
            Mole fractions [-], shape ``(num_species,)``.
        temp : float, optional
            Temperature [K]; seeds a bubble pressure when ``pres`` is None.
        pres : float, optional
            Pressure [Pa]; seeds a bubble or dew temperature.
        used : ndarray of bool, optional
            Species that enter the seed, shape ``(num_species,)``; every
            species when None. The others are not evaluated at all.

        Returns
        -------
        float
            ``sum_i x_i * psat_i`` [Pa] or ``sum_i x_i * Tsat_i`` [K], over
            the used species.

        Notes
        -----
        Initial guess only; it matches the pre-#414 seed so convergence
        paths are unchanged. A species without ``p_vap`` data is evaluated
        with zero Antoine coefficients, exactly as the zero rows of earlier
        releases were: ``Tsat = -0`` K and ``psat = 1`` Pa. A seed is not a
        result; the converged root is checked with ``getKeqVLE``. An unused
        species is left out before evaluation, because its saturation value
        can be infinite (``log10(pres) = A`` makes ``Tsat`` divide by zero)
        and zero times infinity would make the seed NaN.
        """
        p_vap_cts = self._legacy_antoine_rows()  # [-], [K], [K]
        if used is not None:
            mole_frac = np.asarray(mole_frac)[used]  # [-]
            p_vap_cts = p_vap_cts[used]  # [-], [K], [K]
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
                  gamma_model: str = 'ideal', *,
                  weights: Optional[np.ndarray] = None) -> np.ndarray:
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
        weights : ndarray, optional
            Keyword only. Amounts by which the caller weights the returned
            ratios [any basis], with exactly the shape of ``x_liq``
            (``(num_species,)`` or ``(num_points, num_species)``); they
            decide which species need data (see Raises) in place of
            ``x_liq``. A non-ideal activity model is then evaluated on the
            species with a nonzero weight in some row only, with their
            ``x_liq`` entries, as if the others were absent from the
            liquid; the others get the neutral coefficient 1 and need no
            activity parameters (see ``_activity_coefficients``). For
            example, the evaporators pass their structural presence mask
            (``PharmaPy.Presence.SpeciesPresence``), with the batch
            ``Evaporator``'s excluded supercritical species set to zero,
            because they never use those ratios.

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
            If ``gamma_model`` is not a supported, case-sensitive selector,
            if ``temp`` has several entries and ``x_liq`` several rows in
            unequal numbers, or if ``weights`` does not have the shape of
            ``x_liq``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If a species lacks ``p_vap`` coefficients in a row where it is
            at or below its critical temperature (or has no ``t_crit``) and
            has a nonzero ``x_liq`` (or ``weights``) entry, or lacks
            ``henry_constant`` [Pa] in a row where it is above its critical
            temperature and has a nonzero entry. Rows pair as in the
            returned ratios: temperature ``i`` with composition row ``i``, a
            single temperature or composition applying to every row. A
            species present only in rows where it is above ``t_crit`` uses
            its Henry constant there and needs no ``p_vap``, and one present
            only at or below ``t_crit`` needs no Henry constant. Also raised
            if no species supplies ``t_crit`` (naming every species), or if
            no species supplies ``henry_constant`` while some species is
            above its critical temperature in some row, whatever its
            fraction (naming those species), as earlier releases raised
            ``AttributeError`` there; with ``weights``, only species with a
            nonzero weight in such a row count. Also raised, whatever the
            fractions, if ``p_vap`` is stored as one scalar per species
            instead of Antoine coefficients (see ``_property_rows``). With
            ``weights`` and ``gamma_model='UNIQUAC'``, also if a weighted
            species lacks ``ri``, ``qi`` or (when the database has the key)
            ``qip``. UNIFAC group counts are taken as given: no
            species-named check applies to them, and invalid counts of a
            weighted species give non-finite ratios.

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
        exactly zero; it is not a vapor pressure. A zero-weight species
        above its critical temperature whose ``henry_constant`` is NaN keeps
        a NaN ratio there, as in earlier releases: the evaporators and flash
        units index such ratios out or give an absent species a unit ratio.
        Nonzero is bitwise: a round-off-level fraction counts as present. A
        fraction integrated by a solver can carry round-off where it should
        be zero, so the unit operations that decide presence structurally
        (``PharmaPy.Presence``; among them the reactors, the evaporators and
        flash units, and the dynamic distillation column for its plates)
        pass a presence mask as ``weights`` and give an absent species a
        unit ratio, and the shortcut design and the steady distillation
        column pass the presence of the exact compositions they evaluate.
        The user guide (``advanced_usage.rst``, "Missing property data")
        lists the units and their rules; other unit operations need data
        for every species they carry as a state.
        """
        return self._vle_ratios(temp, pres, x_liq, gamma_model, check=True,
                                weights=weights)

    def _vle_ratios(self, temp: Optional[Union[float, np.ndarray]],
                    pres: Optional[float], x_liq: Optional[np.ndarray],
                    gamma_model: str, check: bool,
                    weights: Optional[np.ndarray] = None) -> np.ndarray:
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
            caller then checks with ``getKeqVLE``. The Henry-constant rule
            applies in both cases: a supercritical species with a nonzero
            weight and no ``henry_constant`` raises, and a zero-weight one
            gets a NaN ratio.
        weights : ndarray, optional
            As in ``getKeqVLE``; ``x_liq`` when ``None``.

        Returns
        -------
        ndarray
            Ratios ``y_i/x_i`` [-], shaped as in ``getKeqVLE``.

        Raises
        ------
        ValueError
            If ``gamma_model`` is not a supported selector, the rows of
            ``temp`` and ``x_liq`` cannot pair, or ``weights`` does not have
            the shape of ``x_liq`` (see ``getKeqVLE``).
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            With ``check``, as in ``getKeqVLE``. Also without ``check`` if
            no species supplies ``t_crit``, if no species supplies
            ``henry_constant`` while some species (with ``weights``, some
            weighted species) is supercritical, if a
            weighted supercritical species has no ``henry_constant``, or if
            ``p_vap`` is stored as scalars (see ``_property_rows``).
        """
        validate_activity_model(gamma_model)

        if temp is None:
            temp = self.temp

        if pres is None:
            pres = self.pres

        if x_liq is None:
            x_liq = self.mole_frac

        if weights is not None:
            # weights decide which species need data, so a scalar or
            # mis-shaped array must not silently switch the checks off.
            weights = self._validate_weights_shape(weights, x_liq, 'x_liq')

        # Rows pair temperature i with composition row i.
        num_temps = np.size(temp) if np.ndim(temp) == 1 else 1  # [-]
        num_rows = np.shape(x_liq)[0] if np.ndim(x_liq) == 2 else 1  # [-]
        if num_temps > 1 and num_rows > 1 and num_temps != num_rows:
            raise ValueError(
                "getKeqVLE pairs temperature i with composition row i: temp "
                f"has {num_temps} entries but x_liq has {num_rows} rows; pass "
                "a single temperature, a single composition, or equal counts")

        t_crit = self._require_property(
            't_crit', 'are classified as sub- or supercritical, so their VLE '
            'ratio')  # [K]

        p_vap_cts = self._legacy_antoine_rows()  # [-], [K], [K]
        num_sp = len(self.name_species)  # [-]
        crit = isinstance(temp, np.ndarray) and temp.ndim == 1
        if crit:
            supercrit = temp[:, np.newaxis] > t_crit
        else:
            supercrit = temp > t_crit

        # Rows pair as in the returned ratios; a species needs data only in
        # a row whose weighting entry is nonzero.
        weighted = np.atleast_2d(np.asarray(
            x_liq if weights is None else weights) != 0)  # [-]

        henry = self._scalar_values('henry_constant')  # [Pa], NaN if missing
        if not hasattr(self, 'henry_constant') and np.any(supercrit):
            # Without weights every supercritical species needs the absent
            # Henry data, whatever its fraction, as in earlier releases; with
            # weights only a weighted one does, so a species that the caller
            # marks absent needs none.
            lacking_henry = np.atleast_2d(supercrit)  # [-]
            if weights is not None:
                lacking_henry = lacking_henry & weighted  # [-]
            self._check_species_rows(
                'henry_constant', henry, np.arange(num_sp),
                lacking_henry.any(axis=0),
                'are above t_crit at a requested temperature, so their VLE '
                'ratio', remedy=' or evaluate at or below t_crit')

        p_vap = self._evaluate_antoine(p_vap_cts, temp)  # [Pa]
        if crit:
            if np.any(supercrit):
                p_vap = np.where(supercrit, henry, p_vap)  # [Pa]
        else:
            if any(supercrit):
                p_vap[supercrit] = henry[supercrit]  # [Pa]

        # Antoine data are needed where a species is at or below t_crit,
        # and a Henry constant where it is above, in a weighted row. The
        # message names what made the species needed.
        carry = ('carry a nonzero fraction' if weights is None
                 else 'carry a nonzero weight')
        if check:
            uses_antoine = np.atleast_2d(~supercrit)  # [-]
            needed = (uses_antoine & weighted).any(axis=0)
            self._check_species_rows(
                'p_vap', self._property_rows('p_vap'), np.arange(num_sp),
                needed,
                f'{carry} and are at or below t_crit (or have no t_crit) '
                'at a requested temperature, so their VLE ratio')
        # Unlike the Antoine placeholder, a NaN Henry ratio of a weighted
        # species makes the result NaN, so it is no valid solver iterate
        # either; the rule therefore applies with and without check.
        needs_henry = (np.atleast_2d(supercrit) & weighted).any(axis=0)
        self._check_species_rows(
            'henry_constant', henry, np.arange(num_sp), needs_henry,
            f'{carry} and are above t_crit at a requested temperature, so '
            'their VLE ratio',
            remedy=' or evaluate at or below t_crit')

        # With weights, the activity model sees only the weighted species.
        gamma = self._activity_coefficients(gamma_model, x_liq, temp,
                                            weights)  # [-]

        k_vals = p_vap * gamma / pres

        return k_vals

    def _activity_coefficients(self, gamma_model: str, x_liq: np.ndarray,
                               temp: Union[float, np.ndarray],
                               weights: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """Evaluate liquid activity coefficients, on weighted species only.

        Parameters
        ----------
        gamma_model : {'ideal', 'UNIFAC', 'UNIQUAC'}
            Validated activity model.
        x_liq : ndarray
            Liquid mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        temp : float or ndarray
            Temperature [K], scalar or paired ``(num_points,)`` profile.
        weights : ndarray, optional
            Presence weights [-] with the shape of ``x_liq`` (validated by
            the caller). Without them every species enters the model.

        Returns
        -------
        ndarray
            Activity coefficients [-] with the shape of ``x_liq``; with
            ``weights``, exactly 1 for a species whose weight is zero in
            every row.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. With
            ``weights`` and UNIQUAC, if a weighted species lacks ``ri``,
            ``qi`` or, when the database has the key, ``qip``.

        Notes
        -----
        With ``weights``, the model is evaluated on the species with a
        nonzero weight in some row, as if the others were absent from the
        liquid: their fractions are left out (not renormalized) and their
        parameters, including UNIQUAC interaction rows and columns or
        UNIFAC group counts, are not used, so they need none (issue #425).
        Their neutral coefficient of exactly 1 is part of this contract:
        the VLE ratios do not use it (callers give absent species a unit
        ratio), but the extractors' default distribution ratios
        ``gamma_light / gamma_heavy`` rely on it, so that an absent species
        gets the ratio 1 whatever its data. Callers give zero weight only to
        species whose fraction is zero or solver round-off of zero, so the
        present species' coefficients equal those of the mixture without
        the absent ones.
        """
        if gamma_model == 'ideal':
            return np.ones_like(x_liq)

        if weights is None:
            if gamma_model == 'UNIFAC':
                return self.UNIFAC_DMD(x_liq, temp)

            return self.UNIQUAC(x_liq, temp)

        x_liq = np.asarray(x_liq, dtype=float)  # [-]
        idx = np.flatnonzero(self._needed_by_weights(
            weights, len(self.name_species)))  # weighted species
        gamma = np.ones(x_liq.shape)  # [-], neutral for absent species
        if idx.size == 0:
            return gamma

        if gamma_model == 'UNIQUAC':
            parameters = ('ri', 'qi') + (('qip',) if hasattr(self, 'qip')
                                         else ())
            for prop_name in parameters:
                self._check_species_rows(
                    prop_name, self._scalar_values(prop_name)[idx], idx,
                    np.ones(idx.size, dtype=bool),
                    'carry a nonzero weight, so their UNIQUAC activity '
                    'coefficient')
            gamma[..., idx] = self.UNIQUAC(x_liq[..., idx], temp,
                                           idx=idx)  # [-]
        else:
            gamma[..., idx] = self.UNIFAC_DMD(x_liq[..., idx], temp,
                                              idx=idx)  # [-]

        return gamma

    def UNIQUAC(self, mole_frac: Optional[np.ndarray] = None,
                temp: Optional[Union[float, np.ndarray]] = None,
                idx: Optional[Sequence[int]] = None) -> np.ndarray:
        r""" Calculate activity coefficients :math:`\gamma_i` using UNIQUAC model.

        Parameters
        ------------
        mole_frac : ndarray
            Liquid mole fractions [-], shape (num_species,) or
            (num_points, num_species). Required by the model. With ``idx``,
            the fractions of the selected species only.
        temp : float or ndarray
            Temperature [K], scalar for one composition or a paired
            (num_points,) profile. Required by the model.
        idx : sequence of int, optional
            Database indices of the species that form the mixture, in
            order; the others are left out as if absent, and their
            parameters are not used. Every species when None.
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

        if idx is not None:
            # Only the selected species form the mixture (see idx).
            idx = np.asarray(idx, dtype=int)
            ri = np.asarray(ri, dtype=float).ravel()[idx]  # [-]
            qi = np.asarray(qi, dtype=float).ravel()[idx]  # [-]
            qip = np.asarray(qip, dtype=float).ravel()[idx]  # [-]
            amk = np.asarray(amk, dtype=float)[np.ix_(idx, idx)]  # [J/mol]

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

    def UNIFAC_DMD(self, x_i: Optional[np.ndarray] = None,
                   temp: Optional[Union[float, np.ndarray]] = None,
                   idx: Optional[Sequence[int]] = None) -> np.ndarray:
        """Calculate liquid activity coefficients with UNIFAC-Dortmund.

        Parameters
        ----------
        x_i : ndarray, optional
            Liquid mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``; with ``idx``, of the selected
            species only, in ``idx`` order. The phase composition when None.
        temp : float or ndarray, optional
            Temperature [K]: a scalar for one composition, or shape
            ``(num_points,)`` paired row by row with a two-dimensional
            ``x_i``. The phase temperature when None.
        idx : sequence of int, optional
            Database indices of the species that form the mixture, in
            order; the group counts of the others are not used, as if they
            were absent. Every species when None.

        Returns
        -------
        ndarray
            Activity coefficients [-] with the shape of ``x_i`` (squeezed
            to ``(num_selected,)`` for a one-dimensional ``x_i``).

        Raises
        ------
        AttributeError
            If the database has no UNIFAC groups (``unifac_groups`` and
            ``vk`` in its ``interaction`` record), so the group parameters
            below were never set.

        Notes
        -----
        The model uses instance attributes set from the database at
        construction (``get_UNIFACParams``): group volume and surface
        parameters ``Rk`` and ``Qk`` [-], shape ``(num_groups,)``; group
        interaction coefficients ``a_unifac`` [K], ``b_unifac`` [-] and
        ``c_unifac`` [1/K], shape ``(num_groups, num_groups)``, with
        ``psi_mn = exp(-(a_mn + b_mn T + c_mn T**2) / T)``; and the group
        counts ``vk`` [-], shape ``(num_species, num_groups)``, one row per
        species. ``ln gamma_i`` is the sum of the Dortmund combinatorial
        term (with ``r_i**(3/4)``) and the residual group term [1], [2].
        Group counts are taken as given; invalid counts of a selected
        species give non-finite coefficients.

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
            """Return the group residual log-activity coefficients.

            Parameters
            ----------
            psi_matrix : ndarray
                Group interaction factors ``psi_mn`` [-], shape
                ``(num_groups, num_groups)``.
            theta_vec : ndarray
                Group surface-area fractions [-], shape ``(num_groups,)``.
            q_vec : ndarray
                Group surface parameters ``Qk`` [-], shape
                ``(num_groups,)``.

            Returns
            -------
            ndarray
                ``ln Gamma_k`` [-], shape ``(num_groups,)``.
            """
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
        if idx is not None:
            # Only the selected species form the mixture (see idx).
            v_matrix = np.asarray(v_matrix)[np.asarray(idx, dtype=int)]  # [-]

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
