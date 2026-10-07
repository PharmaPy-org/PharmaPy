#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:13:10 2020

@author: dcasasor
"""

import numpy as np

from PharmaPy.ThermoModule import ThermoPhysicalManager


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
        """
        up, down = prefix_up, prefix_down

        if 'frac' in up and 'frac' in down:
            method_name = 'frac_to_frac'
            if 'mole' in up:
                fun_kwargs = {'mole_frac': composition}
            elif 'mass' in up:
                fun_kwargs = {'mass_frac': composition}

        elif 'frac' in up and 'conc' in down:
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
            Supplies molar masses ``mw`` [g/mol] and ``getDensity``. Dynamic
            profiles require a single-phase thermophysical object (liquid,
            vapor or solid phase or stream).
        composition : numpy.ndarray
            Upstream composition named ``comp_name``: ``mole_frac`` or
            ``mass_frac`` [-], ``mole_conc`` [mol/L], or ``mass_conc``
            [kg/m**3]; shape (num_times, num_species) for a dynamic profile.
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
        Static flows use the stored state of ``matter_object``. Dynamic flows
        evaluate molar mass [g/mol] and mass density [kg/m**3] for every
        sample from its composition and, when supplied, its temperature;
        pressure is the stored state. ``getDensity`` decides the temperature
        dependence: ideal-gas vapor density varies with each sample, while
        the current liquid pure-component densities are constants.
        Concentration profiles (``mole_conc``, ``mass_conc``) contribute only
        their fractions: the liquid mass density is the ideal-mixing value
        of the database pure-component densities, not the upstream total
        concentration. A dynamic sample whose composition is exactly zero
        (a stopped feed) carries no material and converts to zero flow
        without evaluating mixture properties; non-finite samples are
        converted as usual, so invalid values propagate.
        """
        up, down = prefix_up, prefix_down
        stopped = None  # dynamic rows with exactly zero composition

        if np.asarray(flow).ndim == 0:
            mw_av = matter_object.mw_av  # [g/mol]
            density = matter_object.getDensity()  # [kg/m**3]
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
            density_kwargs = {} if temp is None else {'temp': temp}  # [K]
            fed_density = matter_object.getDensity(
                **frac_kwargs, **density_kwargs)  # [kg/m**3], per fed sample

            # Unit placeholders for stopped rows; their flow is zeroed below.
            mw_av = np.ones(len(stopped))  # [g/mol]
            density = np.ones(len(stopped))  # [kg/m**3]
            mw_av[~stopped] = fed_mw_av
            density[~stopped] = fed_density

        # Convert units; 1000 g/kg relates [g/mol] to [kg/mol]
        if 'mass' in up and 'mole' in down:
            flow_out = flow / mw_av * 1000  # mol/s
        elif 'mole' in up and 'mass' in down:
            flow_out = flow * mw_av / 1000  # kg/s
        elif 'vol' in down:
            if 'mole' in up:
                molar_density = density * 1000 / mw_av  # [mol/m**3]
                flow_out = flow / molar_density  # [m**3/s]
            else:
                flow_out = flow / density  # [m**3/s]

        elif 'vol' in up:
            if 'mole' in down:
                molar_density = density * 1000 / mw_av  # [mol/m**3]
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
