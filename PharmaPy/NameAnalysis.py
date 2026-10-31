#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:13:10 2020

@author: dcasasor
"""

import sys
from typing import Optional, Tuple, Union

import numpy as np

from PharmaPy.ThermoModule import ThermoPhysicalManager

# Largest fraction difference [-] still treated as the carrier's own held
# state. A held state's published fractions and the carrier's stored ones
# are the same composition, recomputed at most by a normalization and a
# mass/mole conversion (one sum over the species and a few products per
# entry), whose float64 round-off is a few tens of machine epsilons. Any
# physically different composition differs by many orders more. The
# epsilon is Python's float64 one, equal to ``np.finfo(float).eps``, so the
# constant stays defined when the documentation build mocks NumPy.
HELD_STATE_FRACTION_ATOL = 64 * sys.float_info.epsilon  # [-]


def getBipartiteNames(first, second):
    """
    Create bipartite graph for matching phases. It only matches phase names.
    It should rely on some thermodynamic criterion to decide compatible phases.

    Parameters
    ----------
    first : dict
        name dict of the upstream UO.
    second : dict
        name dict of the downstream UO.

    Returns
    -------
    graph : dict
        bipartite graph.

    """
    graph = {}
    for two in second:
        for one in first:
            if one == second:
                graph[two] = one
                break

    return graph


def getBipartiteMultiPhase(first, second):
    """
    Creates bipartite graph for input dictionaries specifying phases and states

    Parameters
    ----------
    first : dict
        DESCRIPTION.
    second : dict
        DESCRIPTION.

    Returns
    -------
    graph : dict

    """

    upper_graph = getBipartiteNames(first, second)

    graph = {}
    for phase_down, phase_up in upper_graph.items():
        key = '/'.join(phase_down, phase_up)
        graph[key] = getBipartite(first[phase_up], second[phase_down])

    return graph


def getBipartite(first, second):
    graph = {}
    types = {}
    comp_names = ['conc', 'frac']
    amount_names = ['mass', 'moles', 'vol']

    num_first = len(first)
    for two in second:
        for count, one in enumerate(first):
            if 'distr' in one and 'solid_conc' in two:
                graph[two] = one
                types['distrib'] = (one, two)
                break
            elif any(word in one for word in comp_names) and any(word in two for word in comp_names):
                graph[two] = one
                types['composition'] = (one, two)
                break
            elif 'flow' in one and 'flow' in two:
                graph[two] = one
                types['flow'] = (one, two)
                break
            elif 'distr' in one and 'distr' in two:
                graph[two] = one
                types['distrib'] = (one, two)
                break
            elif any(one == word for word in amount_names) and any(two == word for word in amount_names):
                graph[two] = one
                types['amount'] = (one, two)
            elif one == two:
                graph[two] = one
                break
            elif count == num_first - 1:
                graph[two] = None

    return graph, types


def get_types(names):
    conc_type = [name for name in names
                 if ('conc' in name) or ('frac' in name)][0]

    distrib_type = [name if 'distr' in name else None
                    for name in names][0]

    distrib_type = list(filter(lambda x: 'distr' in x, names))

    flow_type = list(filter(lambda x: 'flow' in x, names))

    amount_type = list(
        filter(lambda x: x == 'mass' or x == 'moles' or x == 'vol', names))

    if not distrib_type:
        distrib_type = None
    else:
        distrib_type = distrib_type[0]

    if not flow_type:
        flow_type = None
    else:
        flow_type = flow_type[0]

    if not amount_type:
        amount_type = None
    else:
        amount_type = amount_type[0]

    return conc_type, distrib_type, flow_type, amount_type


def _concentration_densities(composition: np.ndarray, comp_name: str,
                             molar_mass: np.ndarray) -> Tuple[
                                 Union[float, np.ndarray],
                                 Union[float, np.ndarray]]:
    """Return the mass and molar densities that concentrations imply.

    Parameters
    ----------
    composition : numpy.ndarray
        Species molar [mol/L] or mass [kg/m**3] concentrations, shape
        (num_species,) or (num_samples, num_species), in database order.
    comp_name : {'mole_conc', 'mass_conc'}
        Basis of ``composition``.
    molar_mass : numpy.ndarray
        Species molar masses [g/mol], shape (num_species,).

    Returns
    -------
    mass_density : float or numpy.ndarray
        Total mass concentration [kg/m**3], scalar or shape (num_samples,).
    molar_density : float or numpy.ndarray
        Total molar concentration [mol/m**3], with the same shape.

    Notes
    -----
    Authoritative concentrations (issue #439) define the densities
    directly: ``sum_i C_i * MW_i`` [mol/L * g/mol = kg/m**3] and
    ``1000 * sum_i C_i`` [mol/m**3] (1000 L/m**3). For mass
    concentrations ``C_i = mass_conc_i / MW_i`` [mol/L].
    """
    composition = np.asarray(composition, dtype=float)
    if comp_name == 'mole_conc':
        mole_conc = composition  # [mol/L]
        mass_conc = composition * molar_mass  # [kg/m**3]
    else:
        mass_conc = composition  # [kg/m**3]
        mole_conc = composition / molar_mass  # [mol/L]

    mass_density = np.sum(mass_conc, axis=-1)  # [kg/m**3]
    molar_density = np.sum(mole_conc, axis=-1) * 1000  # [mol/m**3]
    return mass_density, molar_density


def _is_held_state(matter_object: object, composition: np.ndarray,
                   comp_name: str) -> bool:
    """Tell whether fractions are a concentration-basis liquid's own state.

    Parameters
    ----------
    matter_object : object
        Transferred matter.
    composition : numpy.ndarray
        Upstream fractions [-], shape (num_species,) for a static source or
        (num_times, num_species) for a profile.
    comp_name : str
        Upstream composition name; only ``'mass_frac'`` and ``'mole_frac'``
        can describe a held state.

    Returns
    -------
    bool
        True if the matter is a liquid with authoritative concentrations
        (``concentration_basis``), the composition is static or has a single
        sample, and that sample equals the matter's stored fractions of the
        same basis within ``HELD_STATE_FRACTION_ATOL`` [-].

    Notes
    -----
    Only then is the matter's inventory density ``sum(mass_conc)``
    [kg/m**3] known to apply to the sample (issue #439); any other
    composition carries no absolute packing and keeps ideal mixing.
    """
    fractions = np.asarray(composition, dtype=float)  # [-]
    if fractions.ndim > 1 and len(fractions) != 1:
        return False

    return bool(held_state_rows(matter_object, fractions, comp_name).all())


def held_state_rows(matter_object: object, composition: np.ndarray,
                    comp_name: str) -> np.ndarray:
    """Flag the composition rows that are a liquid's own held state.

    Parameters
    ----------
    matter_object : object
        Phase or stream carrying the composition.
    composition : numpy.ndarray
        Fractions [-] named ``comp_name``, shape (num_species,) or
        (num_rows, num_species).
    comp_name : str
        Composition name; only ``'mass_frac'`` and ``'mole_frac'`` can
        describe a held state.

    Returns
    -------
    numpy.ndarray of bool
        Shape (1,) for a one-dimensional composition, else (num_rows,).
        True where the matter is a liquid with authoritative concentrations
        (``concentration_basis``) and the row equals its stored fractions of
        the same basis within ``HELD_STATE_FRACTION_ATOL`` [-].

    Notes
    -----
    Such rows are the matter's own state, so its inventory density
    ``getInventoryDensity()`` [kg/m**3] applies to them (issue #439); any
    other row carries no absolute packing and keeps ideal mixing.
    """
    matches = _stored_fraction_rows(matter_object, composition, comp_name)
    if not getattr(matter_object, 'concentration_basis', False):
        return np.zeros_like(matches)

    return matches


def _stored_fraction_rows(matter_object: object, composition: np.ndarray,
                          comp_name: str) -> np.ndarray:
    """Flag the composition rows equal to the matter's stored fractions.

    Parameters
    ----------
    matter_object : object
        Phase or stream carrying the composition.
    composition : numpy.ndarray
        Composition named ``comp_name``, shape (num_species,) or
        (num_rows, num_species).
    comp_name : str
        Composition name; rows can match only for ``'mass_frac'`` and
        ``'mole_frac'`` [-].

    Returns
    -------
    numpy.ndarray of bool
        Shape (1,) for a one-dimensional composition, else (num_rows,).
        True where the row equals the matter's stored fractions of the same
        basis within ``HELD_STATE_FRACTION_ATOL`` [-].
    """
    fractions = np.atleast_2d(np.asarray(composition, dtype=float))  # [-]
    stored = getattr(matter_object, comp_name, None)
    if comp_name not in ('mass_frac', 'mole_frac') or stored is None:
        return np.zeros(len(fractions), dtype=bool)

    stored = np.asarray(stored, dtype=float)  # [-]
    if stored.shape != fractions.shape[1:]:
        return np.zeros(len(fractions), dtype=bool)

    return np.all(np.abs(fractions - stored) <= HELD_STATE_FRACTION_ATOL,
                  axis=1)


def _held_state_concentration(prefix_up: str, prefix_down: str,
                              composition: np.ndarray,
                              matter_object: object) -> Optional[np.ndarray]:
    """Convert a liquid's held fractions back to its own concentrations.

    Parameters
    ----------
    prefix_up : {'mole_frac', 'mass_frac'}
        Basis of ``composition`` [-].
    prefix_down : {'mole_conc', 'mass_conc'}
        Requested concentration basis, [mol/L] or [kg/m**3].
    composition : numpy.ndarray
        Species fractions [-], shape (num_species,) for a static source or
        (num_times, num_species) for a profile, in database order.
    matter_object : object
        Transferred matter.

    Returns
    -------
    numpy.ndarray or None
        Concentrations in the ``prefix_down`` basis with the shape of
        ``composition``, or None when the composition is not the matter's
        held state (see ``_is_held_state``).

    Notes
    -----
    A static or single-sample composition equal to the stored fractions of
    a liquid with authoritative concentrations is that liquid's held state
    (issue #439), so the liquid's inventory density
    ``getInventoryDensity()`` [kg/m**3] applies:
    ``mass_conc_i = w_i * rho_inv`` [kg/m**3] and
    ``C_i = mass_conc_i / MW_i`` [mol/L]. Fractions carry no absolute
    packing, so any other composition keeps the ideal-mixing
    ``frac_to_conc`` conversion.
    """
    if not _is_held_state(matter_object, composition, prefix_up):
        return None

    fractions = np.asarray(composition, dtype=float)  # [-]
    if 'mole' in prefix_up:
        mass_frac = matter_object.frac_to_frac(mole_frac=fractions)  # [-]
    else:
        mass_frac = fractions  # [-]

    mass_conc = mass_frac * matter_object.getInventoryDensity()  # [kg/m**3]
    if 'mass' in prefix_down:
        return mass_conc

    return mass_conc / matter_object.mw  # [mol/L]


def get_dict_states(names, num_species, num_distr, states):
    count = 0
    dict_states = {}

    for name in names:
        if 'conc' in name or 'frac' in name:
            idx_composition = range(count, count + num_species)
            dict_states[name] = states.T[idx_composition].T

            count += num_species
        elif 'distrib' in name or 'mu_n' in name:
            idx_distrib = range(count, count + num_distr)
            dict_states[name] = states.T[idx_distrib].T

            count += num_distr
        else:
            dict_states[name] = states.T[count].T

            count += 1

    return dict_states


class NameAnalyzer:
    def __init__(self, names_up, names_down, num_species, num_distr=None):
        self.names_up = names_up
        self.names_down = names_down

        self.num_species = num_species
        self.num_distr = num_distr

        self.bipartite, self.conv_types = getBipartite(names_up, names_down)

    def get_idx(self):
        count = 0

        idx_flow = None
        idx_amount = None
        idx_distrib = None

        for name in self.names_up:
            if 'conc' in name or 'frac' in name:
                idx_composition = range(count, count + self.num_species)

                count += self.num_species
            elif 'distrib' in name:
                idx_distrib = range(count, count + self.num_distr)

                count += self.num_distr
            else:
                if 'flow' in name:
                    idx_flow = count
                elif name == 'mass' or name == 'moles' or name == 'vol':
                    idx_amount = count

                count += 1

        return idx_composition, idx_flow, idx_amount, idx_distrib

    def convertUnits(self, matter_transf):
        """Convert upstream states to the names requested downstream.

        Parameters
        ----------
        matter_transf : LiquidStream, VaporStream or similar phase object
            Transferred matter whose ``y_upstream`` maps upstream state names
            to scalars (static source) or profiles with time on the first
            axis: flows of shape (num_times,) in [kg/s], [mol/s] or
            [m**3/s], compositions of shape (num_times, num_species), and
            optional temperature ``temp`` [K] of shape (num_times,). Its
            thermophysical methods supply molar masses and densities.

        Returns
        -------
        dict
            Downstream state names mapped to converted values with the same
            time axis and species order as the upstream profiles. Names
            matching upstream are passed through unchanged.
        """
        dict_in = matter_transf.y_upstream

        dict_out = {}

        for target, source in self.bipartite.items():
            if source is not None:

                if target != source:
                    y_j = dict_in[source]
                    
                    if 'distrib' in target or 'solid_conc' in target:
                        converted_state = self.__convert_distrib(
                            source, target, y_j, matter_transf)

                    elif 'conc' in target or 'frac' in target:
                        converted_state = self.__convertComposition(
                            source, target, y_j, matter_transf)

                    elif 'flow' in target:
                        comp = self.conv_types['composition']
                        
                        converted_state = self.__convertFlow(
                            source, target, y_j, matter_transf,
                            dict_in[comp[0]], comp[0],
                            temp=dict_in.get('temp'))

                    dict_out[target] = converted_state

                else:
                    dict_out[target] = dict_in[source]

        return dict_out

    def __convertComposition(self, prefix_up, prefix_down, composition,
                             matter_object):
        """Convert a composition between fraction and concentration bases.

        Parameters
        ----------
        prefix_up, prefix_down : str
            Upstream and downstream composition names: ``'mole_frac'`` or
            ``'mass_frac'`` [-], ``'mole_conc'`` [mol/L] or ``'mass_conc'``
            [kg/m**3].
        composition : numpy.ndarray
            Upstream composition, shape (num_species,) or
            (num_times, num_species), in database species order.
        matter_object : phase, stream or slurry object
            Supplies the conversion method, directly or through one of its
            ``Phases``.

        Returns
        -------
        numpy.ndarray
            Composition in the ``prefix_down`` basis with the shape and
            species order of ``composition``.

        Notes
        -----
        Fractions are converted to concentrations with the ideal-mixing
        density of each composition (``frac_to_conc``), except for a static
        or single-sample composition equal to the stored fractions of a
        liquid with authoritative concentrations, such as the
        ``BatchToFlowConnector`` discharge: that is the liquid's held state
        (``_is_held_state``), converted with its inventory density so
        its concentrations are recovered (issue #439; see
        ``_held_state_concentration``). Concentrations are converted to
        fractions by normalization.
        """
        up, down = prefix_up, prefix_down

        if 'frac' in up and 'frac' in down:
            method_name = 'frac_to_frac'
            if 'mole' in up:
                fun_kwargs = {'mole_frac': composition}
            elif 'mass' in up:
                fun_kwargs = {'mass_frac': composition}

        elif 'frac' in up and 'conc' in down:
            held_conc = _held_state_concentration(up, down, composition,
                                                  matter_object)
            if held_conc is not None:
                return held_conc

            method_name = 'frac_to_conc'

            if 'mole' in up:
                fun_kwargs = {'mole_frac': composition}
            elif 'mass' in up:
                fun_kwargs = {'mass_frac': composition}

            if 'mass' in down:
                fun_kwargs['basis'] = 'mass'

        elif 'mole_conc' in up and 'frac' in down:
            method_name = 'conc_to_frac'
            fun_kwargs = {'conc': composition}

            if 'mass' in down:
                fun_kwargs['basis'] = 'mass'
            else:
                fun_kwargs['basis'] = 'mole'

        elif 'mass_conc' in up and 'frac' in down:
            method_name = 'mass_conc_to_frac'
            # An explicit basis returns one array; None would return both.
            fun_kwargs = {'conc': composition,
                          'basis': 'mole' if 'mole' in down else 'mass'}

        elif 'conc' in up and 'conc' in down:
            method_name = 'conc_to_conc'

            if 'mole' in up:
                fun_kwargs = {'mole_conc': composition}
            else:
                fun_kwargs = {'mass_conc': composition}

        method = getattr(matter_object, method_name, None)
        if method is None:
            for phase in matter_object.Phases:
                if hasattr(phase, method_name):
                    method = getattr(phase, method_name)

                    break

        output_composition = method(**fun_kwargs)

        return output_composition

    def __convertFlow(self, prefix_up, prefix_down, flow, matter_object,
                      composition, comp_name, temp=None):
        """Convert a flow between mass, molar, and volume bases.

        Parameters
        ----------
        prefix_up, prefix_down : str
            Upstream and downstream flow names: ``'mass_flow'`` [kg/s],
            ``'mole_flow'`` [mol/s] or ``'vol_flow'`` [m**3/s].
        flow : float or numpy.ndarray
            Upstream flow in the ``prefix_up`` units, scalar for a static
            source or shape (num_times,) for a dynamic profile.
        matter_object : phase or stream object
            Supplies molar masses ``mw`` [g/mol], ``getDensity`` and, for a
            liquid, ``concentration_basis`` and ``getInventoryDensity``.
            Dynamic profiles require a single-phase thermophysical object
            (liquid, vapor or solid phase or stream).
        composition : numpy.ndarray
            Upstream composition named ``comp_name``: ``mole_frac`` or
            ``mass_frac`` [-], ``mole_conc`` [mol/L], or ``mass_conc``
            [kg/m**3]; shape (num_species,) for a static source or
            (num_times, num_species) for a dynamic profile.
        comp_name : str
            Name of the upstream composition state.
        temp : float or numpy.ndarray, optional
            Upstream temperature [K], scalar or shape (num_times,). Used
            only for dynamic profiles; None keeps the stored temperature.

        Returns
        -------
        float or numpy.ndarray
            Flow in the ``prefix_down`` units, with the shape of ``flow``.

        Raises
        ------
        NotImplementedError
            If a dynamic profile is carried by an object that is not a
            single-phase thermophysical object, such as a slurry, whose
            density is per phase rather than per mixture sample.
        ValueError
            If a dynamic profile uses an unsupported composition basis.

        Notes
        -----
        Mass and molar flows are related by the average molar mass [g/mol]
        of the composition: the stored state's for a static flow, each
        sample's for a dynamic profile. Volume flows need a density, chosen
        as follows (issue #439):

        * Concentrations (``mole_conc``, ``mass_conc``) carried by a
          single-phase object are authoritative: the mass density is
          ``sum(mass_conc)`` [kg/m**3] and the molar density
          ``sum(mole_conc) * 1000`` [mol/m**3] of the static composition or
          of each sample, so ``mole_flow = sum(C) * vol_flow * 1000``. No
          density model is used. Earlier releases used the ideal-mixing
          density of the normalized fractions instead, which differs unless
          the concentrations close ideal mixing.
        * Fractions that are the held state of a liquid with authoritative
          concentrations (static or single-sample, equal to its stored
          fractions; see ``_is_held_state``), such as the discharge
          published by ``BatchToFlowConnector``, use that liquid's
          ``getInventoryDensity``, ``sum(mass_conc)`` [kg/m**3].
        * Other static fractions equal to the stored fractions of the
          transferred matter use its stored state, ``getDensity()`` and
          ``mw_av``, as before. Static fractions of another composition use
          that composition's own molar mass and ``getDensity`` at the
          stored temperature and pressure; earlier releases applied the
          stored composition's values to them.
        * Any other dynamic fraction profile has no information on absolute
          packing: each sample uses ``getDensity`` of its own composition
          and, when supplied, temperature. That is the ideal-mixing density
          of the database pure-component liquid densities (constants) for a
          liquid, and the ideal-gas density of each sample for a vapor.

        A dynamic sample whose composition is exactly zero (a stopped feed)
        carries no material and converts to zero flow without evaluating
        mixture properties; non-finite samples are converted as usual, so
        invalid values propagate.
        """
        up, down = prefix_up, prefix_down
        stopped = None  # dynamic rows with exactly zero composition
        from_concentration = (
            comp_name in ('mole_conc', 'mass_conc')
            and ('vol' in up or 'vol' in down)
            and isinstance(matter_object, ThermoPhysicalManager))

        static_other = (
            np.asarray(flow).ndim == 0
            and comp_name in ('mass_frac', 'mole_frac')
            and isinstance(matter_object, ThermoPhysicalManager)
            and not _stored_fraction_rows(matter_object, composition,
                                          comp_name).all())

        if static_other:
            # Static fractions of another composition than the stored one:
            # its own molar mass and ideal-mixing density (see Notes).
            fractions = np.asarray(composition, dtype=float)  # [-]
            if comp_name == 'mole_frac':
                mole_frac = fractions  # [-]
            else:
                mole_frac = matter_object.frac_to_frac(mass_frac=fractions)  # [-]
            mw_av = np.dot(mole_frac, matter_object.mw)  # [g/mol]
            density = matter_object.getDensity(
                **{comp_name: fractions})  # [kg/m**3]
            molar_density = density * 1000 / mw_av  # [mol/m**3]
        elif np.asarray(flow).ndim == 0:
            mw_av = matter_object.mw_av  # [g/mol]
            if from_concentration:
                density, molar_density = _concentration_densities(
                    composition, comp_name, matter_object.mw)
                # [kg/m**3], [mol/m**3]
            elif _is_held_state(matter_object, composition, comp_name):
                density = matter_object.getInventoryDensity()  # [kg/m**3]
                molar_density = density * 1000 / mw_av  # [mol/m**3]
            else:
                density = matter_object.getDensity()  # [kg/m**3]
                molar_density = density * 1000 / mw_av  # [mol/m**3]
        else:
            if not isinstance(matter_object, ThermoPhysicalManager):
                raise NotImplementedError(
                    f"Dynamic '{up}' to '{down}' conversion with "
                    f"'{comp_name}' needs a per-sample mixture density, "
                    f"which {type(matter_object).__name__} does not provide; "
                    "supply a single-phase liquid or vapor stream.")

            # Mixture properties are undefined for stopped samples; evaluate
            # them only for the other rows and give stopped rows zero flow.
            composition = np.asarray(composition, dtype=float)
            held_state = _is_held_state(matter_object, composition,
                                        comp_name)
            stopped = np.all(composition == 0, axis=-1)
            if np.ndim(temp) > 0:
                temp = np.asarray(temp, dtype=float)[~stopped]  # [K]
            composition = composition[~stopped]

            # Per-sample fractions [-], rows normalized for concentrations
            if comp_name == 'mole_frac':
                mole_frac = composition
                frac_kwargs = {'mole_frac': mole_frac}
            elif comp_name == 'mole_conc':
                mole_frac = composition / composition.sum(axis=1,
                                                          keepdims=True)
                frac_kwargs = {'mole_frac': mole_frac}
            elif comp_name in ('mass_frac', 'mass_conc'):
                mass_frac = composition
                if comp_name == 'mass_conc':
                    mass_frac = composition / composition.sum(axis=1,
                                                              keepdims=True)
                mole_frac = matter_object.frac_to_frac(mass_frac=mass_frac)
                frac_kwargs = {'mass_frac': mass_frac}
            else:
                raise ValueError(
                    f"Dynamic flow conversion does not support the "
                    f"'{comp_name}' composition basis of "
                    f"{type(matter_object).__name__}; use mole_frac, "
                    "mass_frac, mole_conc or mass_conc.")

            fed_mw_av = np.dot(mole_frac, matter_object.mw)  # [g/mol]
            if from_concentration:
                fed_density, fed_molar_density = _concentration_densities(
                    composition, comp_name, matter_object.mw)
                # [kg/m**3], [mol/m**3], per fed sample
            else:
                if held_state:
                    # The liquid's own held state, see Notes [kg/m**3]
                    fed_density = matter_object.getInventoryDensity()
                else:
                    density_kwargs = ({} if temp is None
                                      else {'temp': temp})  # [K]
                    # [kg/m**3], per fed sample
                    fed_density = matter_object.getDensity(
                        **frac_kwargs, **density_kwargs)
                # [mol/m**3], 1000 g/kg
                fed_molar_density = fed_density * 1000 / fed_mw_av

            # Unit placeholders for stopped rows; their flow is zeroed below.
            mw_av = np.ones(len(stopped))  # [g/mol]
            density = np.ones(len(stopped))  # [kg/m**3]
            molar_density = np.ones(len(stopped))  # [mol/m**3]
            mw_av[~stopped] = fed_mw_av
            density[~stopped] = fed_density
            molar_density[~stopped] = fed_molar_density

        # Convert units; 1000 g/kg relates [g/mol] to [kg/mol]
        if 'mass' in up and 'mole' in down:
            flow_out = flow / mw_av * 1000  # mol/s
        elif 'mole' in up and 'mass' in down:
            flow_out = flow * mw_av / 1000  # kg/s
        elif 'vol' in down:
            if 'mole' in up:
                flow_out = flow / molar_density  # [m**3/s]
            else:
                flow_out = flow / density  # [m**3/s]

        elif 'vol' in up:
            if 'mole' in down:
                flow_out = flow * molar_density  # [mol/s]
            else:
                flow_out = flow * density  # [kg/s]

        if stopped is not None and stopped.any():
            flow_out = np.where(stopped, 0.0, flow_out)  # stopped feed

        return flow_out

    def __convert_distrib(self, prefix_up, prefix_down, distrib,
                          matter_object):
        up, down = prefix_up, prefix_down

        if 'distrib' in up and 'total' in down:
            out = distrib
        elif 'num' in up and 'vol' in down:
            pass
        elif up == 'distrib' and down == 'solid_conc':
            out = matter_object.getSolidsConcentr(distrib=distrib,
                                                  basis='mass')

        return out
