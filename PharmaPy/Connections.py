# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:36:35 2020

@author: dcasasor
"""

from collections import deque
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from PharmaPy.NameAnalysis import NameAnalyzer, get_dict_states
from PharmaPy.Interpolation import local_newton_interpolation

from scipy.interpolate import CubicSpline

import numpy as np
import copy


def interpolate_inputs(time: Union[float, np.ndarray],
                       t_inlet: Union[float, np.ndarray],
                       y_inlet: np.ndarray,
                       **kwargs_interp_fn) -> Union[float, np.ndarray]:
    """Interpolate upstream profiles, holding the endpoint after their support.

    Parameters
    ----------
    time : float or numpy.ndarray
        Evaluation time [s], scalar or an array of shape (num_times,) in any
        order.
    t_inlet : float or numpy.ndarray
        Upstream times [s], shape (num_upstream_times,); a scalar represents
        one sample and requires exactly one row in ``y_inlet``.
    y_inlet : numpy.ndarray
        Values with time on the first axis. Multiple upstream samples support
        scalar fields of shape (num_upstream_times,) and vector fields of
        shape (num_upstream_times, num_fields), preserving their physical
        units (for example mass flow [kg/s] or mass fractions [-]). A single
        upstream sample may have additional field axes, which are preserved.
    **kwargs_interp_fn : dict
        Keyword arguments forwarded to the selected interpolation routine.

    Returns
    -------
    scalar or numpy.ndarray
        Interpolated values in the input units. Scalar time removes the time
        axis; array time replaces it with the requested time axis, whose rows
        follow the caller's query order.

    Raises
    ------
    ValueError
        If a single upstream time is not paired with exactly one value row.

    Notes
    -----
    A single upstream sample is constant for all scalar or array times.
    Multi-point profiles use local Newton interpolation for scalar time and
    cubic splines for array time; values at or beyond the final time are the
    final upstream sample ``y_inlet[-1]`` (in a floating dtype), not a
    polynomial evaluation that could lose it to round-off or node-window
    choice. Earlier times extrapolate, so callers requiring measured support
    must validate the beginning of their evaluation interval.
    """
    t_inlet = np.atleast_1d(t_inlet)  # [s], scalar times represent one sample
    if len(t_inlet) == 1:
        if np.ndim(y_inlet) == 0 or len(y_inlet) != 1:
            raise ValueError(
                'y_inlet must have exactly one value row when t_inlet has '
                'one sample; supply matching time and value rows.')
        if np.ndim(time) == 0:
            return y_inlet[0]
        return np.broadcast_to(
            y_inlet[0], (len(time),) + np.shape(y_inlet)[1:]).copy()

    if np.ndim(time) == 0:
        if time >= t_inlet[-1]:
            # Steady-state hold: return the measured terminal sample itself;
            # astype copies an array row and keeps 1-D profiles scalar.
            samples = np.asarray(y_inlet)
            return samples[-1].astype(np.result_type(samples.dtype, np.float64))

        y_interp = local_newton_interpolation(time, t_inlet, y_inlet,
                                              **kwargs_interp_fn)
    else:
        time = np.asarray(time)  # [s], caller query order
        interpol = CubicSpline(t_inlet, y_inlet, **kwargs_interp_fn)
        flags_hold = time >= t_inlet[-1]

        # Evaluate rows in place so the output keeps the query order, then
        # write the final upstream sample at and after the end of support.
        y_interp = interpol(np.minimum(time, t_inlet[-1]))
        y_interp[flags_hold] = y_inlet[-1]

    return y_interp


def get_input_dict(input_data, name_dict):

    dict_out = {}
    count = 0
    for phase, states in name_dict.items():
        names = list(states.keys())
        lens = list(states.values())

        acum_len = np.cumsum(lens)
        if len(acum_len) > 1:
            acum_len = acum_len[:-1]

        if input_data.ndim == 1:
            splitted = np.split(input_data[count:], acum_len, axis=0)
            splitted = [ar[0] if len(ar) == 1 else ar for ar in splitted]
        else:
            splitted = np.split(input_data[:, count:], acum_len, axis=1)

            for ind, val in enumerate(splitted):
                if val.shape[1] == 1:
                    splitted[ind] = val.flatten()

        for ind, elem in enumerate(splitted):
            if isinstance(elem, np.ndarray) and len(elem) == 0:
                splitted[ind] = np.zeros(lens[ind])

        count += acum_len[-1]

        dict_out[phase] = dict(zip(names, splitted))

    return dict_out


def get_missing_field(dim_state, n_times):
    if n_times == 1:
        if dim_state == 1:
            out = 0
        else:
            out = np.zeros(dim_state)
    elif n_times > 1:
        if dim_state == 1:
            out = np.zeros(n_times)

        else:
            out = np.zeros((n_times, dim_state))

    return out


def get_remaining_states(dict_states_in, stream, inlets, time):
    di_out = {}
    time = np.atleast_1d(time)
    for phase, di in dict_states_in.items():
        di_out[phase] = {}
        if 'inlet' in phase.lower():
            for state in di:
                if state not in inlets[phase]:
                    field = getattr(stream, state, None)

                    if field is None:
                        field = get_missing_field(
                            di[state], len(time))

                    elif len(time) > 1:
                        field = np.outer(np.ones_like(time), field)
                        if field.shape[1] == 1:
                            field = field.flatten()

                    di_out[phase][state] = field
        else:
            for state in di:
                if state not in inlets[phase]:
                    sub_phase = getattr(stream, phase)
                    field = getattr(sub_phase, state, None)

                    if field is None:
                        field = get_missing_field(
                            di[state], len(time))

                    di_out[phase][state] = field
    return di_out


def _map_dynamic_inputs(dynamic_inputs: dict, dict_states_in: dict) -> dict:
    """Assign stream-level dynamic fields to the declared inlet groups.

    Parameters
    ----------
    dynamic_inputs : dict
        Field names mapped to the values returned by
        ``DynamicInput.evaluate_inputs``, in the units and shapes produced by
        each control callable (for example volumetric flow [m**3/s] or
        liquid mass concentrations [kg/m**3]).
    dict_states_in : dict
        Destination inlet layout: group keys (``'Inlet'`` for stream-level
        fields, or a phase name such as ``'Liquid_1'``) mapped to containers
        of the field names declared for that group.

    Returns
    -------
    dict
        One entry per layout group, in layout order. Each value holds the
        dynamic fields assigned to that group in their original order and
        with the unchanged values, so no unit or basis conversion occurs.

    Raises
    ------
    ValueError
        If a dynamic field is declared in more than one group, or if it is
        declared in none and the layout has no ``'Inlet'`` group.

    Notes
    -----
    A field declared by exactly one group is assigned to that group. A field
    declared by no group is assigned to ``'Inlet'``, which reproduces the
    established single-group behaviour where every dynamic field is passed
    through under ``'Inlet'``. A stream-level ``DynamicInput`` keys its
    fields by plain name only, so a name shared by several groups (for
    example ``temp`` declared for both ``'Inlet'`` and ``'Liquid_1'``) cannot
    be attributed to one of them; it is rejected instead of being copied to
    every group or overwriting one group's value with another's.
    """
    mapped = {group: {} for group in dict_states_in}

    for name, value in dynamic_inputs.items():
        groups = [group for group, names in dict_states_in.items()
                  if name in names]

        if len(groups) > 1:
            raise ValueError(
                f"DynamicInput field '{name}' is declared by several inlet "
                f"groups {groups} of the destination unit; a stream-level "
                "DynamicInput cannot identify which one it controls. Remove "
                f"the '{name}' control or use a destination whose inlet "
                "layout declares it once.")
        elif groups:
            target = groups[0]
        elif 'Inlet' in dict_states_in:
            target = 'Inlet'
        else:
            raise ValueError(
                f"DynamicInput field '{name}' is not declared by any inlet "
                f"group {list(dict_states_in)} of the destination unit, and "
                "the layout has no 'Inlet' group to receive stream-level "
                "fields.")

        mapped[target][name] = value

    return mapped


def get_inputs_new(time: Union[float, np.ndarray], stream: object,
                   dict_states_in: dict, **kwargs_interp) -> dict:
    """Resolve a destination unit's inlet fields from one stream.

    Parameters
    ----------
    time : float or numpy.ndarray
        Evaluation time [s], scalar or shape (num_times,).
    stream : PharmaPy stream
        Inlet stream. Its ``DynamicInlet``, connected upstream profile
        (``y_upstream``/``y_inlet``/``time_upstream``) or static attributes
        supply the values, in that order of precedence.
    dict_states_in : dict
        Destination inlet layout: group keys mapped to dictionaries of
        declared field names and their dimensions. ``'Inlet'`` (or any key
        containing ``'inlet'``) holds stream-level fields read from
        ``stream``; other keys name a phase attribute of ``stream`` such as
        ``'Liquid_1'``, whose static fields are read from that phase.
    **kwargs_interp : keyword arguments
        Arguments for the interpolation of connected upstream profiles
        only. They are not supported with a ``DynamicInlet``, whose
        ``evaluate_inputs`` accepts the evaluation time alone.

    Returns
    -------
    inputs : dict
        One entry per layout group, each mapping field names to values in the
        units and basis of their source (for example [m**3/s], [K],
        [kg/m**3]). Fields absent from every source are zero-filled.

    Raises
    ------
    ValueError
        If a ``DynamicInlet`` field cannot be assigned to exactly one layout
        group (see Notes).

    Notes
    -----
    ``DynamicInput`` fields are keyed by plain field name. Each one is
    assigned to the single layout group that declares it; fields declared
    by no group go to ``'Inlet'``, so single-group ``{'Inlet': ...}``
    layouts receive every dynamic field exactly as before. A name declared
    by several groups is rejected rather than duplicated or overwritten.
    Declared fields that no control supplies keep their static values from
    the stream (``'Inlet'``) or from the corresponding phase.
    """

    if stream.DynamicInlet is not None:
        inputs = stream.DynamicInlet.evaluate_inputs(time, **kwargs_interp)
        inputs = _map_dynamic_inputs(inputs, dict_states_in)

    elif stream.y_upstream is not None and stream.time_upstream is not None:
        t_inlet = stream.time_upstream
        y_inlet = stream.y_inlet

        ins = {}
        for key, val in y_inlet.items():
            ins[key] = interpolate_inputs(time, t_inlet, val,
                                          **kwargs_interp)

        inputs = {}
        for phase, names in dict_states_in.items():
            inputs[phase] = {}
            for key, vals in ins.items():
                if key in names:
                    inputs[phase][key] = vals

    elif stream.y_upstream is not None:
        inputs = {}

        for phase, names in dict_states_in.items():
            inputs[phase] = {}
            for key, vals in stream.y_inlet.items():
                if key in names:
                    if isinstance(time, np.ndarray) and len(time) > 1:
                        vals = np.outer(np.ones_like(time), vals)

                        if vals.shape[1] == 1:
                            vals = vals.flatten()

                    inputs[phase][key] = vals

    else:
        inputs = {obj: {} for obj in dict_states_in.keys()}

    remaining = get_remaining_states(dict_states_in, stream, inputs, time)

    for key in dict_states_in:
        inputs[key] = {**inputs[key], **remaining[key]}

    return inputs


def get_inputs(time, uo, num_species, num_distr=0):
    Inlet = getattr(uo, 'Inlet', None)

    names_upstream = uo.names_upstream
    names_states_in = uo.names_states_in
    bipartite = uo.bipartite
    if Inlet is None:
        input_dict = {}

        return input_dict

    elif Inlet.y_upstream is None or len(Inlet.y_upstream) == 1:
        # this internally calls the DynamicInput object if not None
        input_dict = Inlet.evaluate_inputs(time)

        for name in names_states_in:
            if name not in input_dict.keys():
                val = getattr(Inlet, name, None)
                if val is None:  # search in subphases inside Inlet
                # if hasattr(uo, 'states_in_phaseid'):
                    obj_id = uo.states_in_phaseid[name]
                    instance = getattr(Inlet, obj_id)
                    val = getattr(instance, name)

                input_dict[name] = val

    else:
        all_inputs = Inlet.InterpolateInputs(time)
        input_upstream = get_dict_states(names_upstream, num_species,
                                         num_distr, all_inputs)

        input_dict = {}
        for key in names_states_in:
            val = input_upstream.get(bipartite[key])
            if val is None:
                val = uo.input_defaults[key]

            input_dict[key] = val

    return input_dict


def topological_bfs(graph: Mapping[str, Sequence[str]]
                    ) -> Tuple[Dict[str, int], List[str]]:
    """Order flowsheet units by deterministic first-in, first-out Kahn BFS.

    Parameters
    ----------
    graph : mapping of str to sequence of str
        Directed flowsheet adjacency. Keys are unit names in declaration
        order; each value lists the unit's successors in an ordered sequence
        such as a list. Units that appear only as successors are nodes too.

    Returns
    -------
    in_degree : dict of str to int
        Residual incoming-edge counters [-] keyed by every node, in order of
        first appearance (each key, then its new successors). Counters are
        decremented as predecessors are visited, so all are zero for an
        acyclic graph; nonzero counters mark nodes on or downstream of a
        cycle. They are not raw graph degrees.
    path : list of str
        Visited nodes in execution order. Shorter than the node count when
        the graph contains a cycle.

    Raises
    ------
    TypeError
        If a successor collection is a set or frozenset, whose iteration
        order depends on string hashing and would make execution order
        irreproducible.

    Notes
    -----
    Ready nodes leave a first-in, first-out queue. The queue is seeded with
    the nodes without predecessors in ``graph`` key order. Visiting a node
    decrements its successors' counters in that node's successor order, and
    a successor joins the queue when its counter reaches zero, that is, after
    its last predecessor is visited. The order is therefore independent of
    ``PYTHONHASHSEED``.

    ``SimulationExec`` executes units in ``path`` order and transfers each
    unit's outlet to its successors right after it runs, so a ``Mixer``
    receives its inlets in the execution order of its predecessors. That is
    declaration order when all of them are seeded sources, but not in
    general: for ``{'S1': ['P'], 'P': ['M'], 'S2': ['M'], 'M': []}`` the
    order is ``S1, S2, P, M`` and ``M`` receives the ``S2`` inlet before the
    ``P`` inlet although ``P`` is declared first.
    """
    in_degree = {}  # [-], incoming-edge counters
    for node, neighbors in graph.items():
        if isinstance(neighbors, (set, frozenset)):
            raise TypeError(
                f"Successors of flowsheet unit {node!r} must be an ordered "
                "sequence such as a list, not a set, so that execution order "
                "is reproducible.")

        in_degree.setdefault(node, 0)
        for neighbor in neighbors:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    path = []

    ready = deque(node for node, count in in_degree.items() if count == 0)

    while ready:
        node = ready.popleft()
        path.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1

            if in_degree[neighbor] == 0:
                ready.append(neighbor)

    return in_degree, path


def convert_str_flowsheet(flowsheet):
    seq = [a.strip() for a in flowsheet.split('-->')]

    out = {}
    num_uos = len(seq)
    for ind in range(num_uos - 1):
        out[seq[ind]] = [seq[ind + 1]]

    out[seq[num_uos - 1]] = []
    return out


class Connection:
    def __init__(self, source_uo, destination_uo):

        self.source_uo = source_uo
        self.destination_uo = destination_uo

    def transfer_data(self):
        self.FeedConnection()
        self.ConvertUnits()
        self.PassPhases()

    def FeedConnection(self):
        self.Matter = self.source_uo.Outlet
        if isinstance(self.Matter, dict):
            self.Matter = self.Matter[self.source_uo.default_output]

        self.num_species = self.Matter.num_species

        self.Matter.y_upstream = self.source_uo.outputs

        time_prof = self.source_uo.result.time

        if self.source_uo.is_continuous:
            self.Matter.time_upstream = time_prof
        else:
            self.Matter.time_upstream = time_prof[-1]

    def _collector_input_names(self) -> list:
        """Return the DynamicCollector input names for the transferred matter.

        Returns
        -------
        list of str
            The collector's crystallizer names (liquid ``mass_conc``
            [kg/m**3], ``vol_flow`` [m**3/s], ``temp`` [K], ``distrib``
            [#/m**3/um], ``mu_n`` [m**n/m**3]) for mixed-phase matter,
            otherwise its liquid-mixer names (``mass_frac`` [-],
            ``mass_flow`` [kg/s], ``temp`` [K]).

        Notes
        -----
        The choice follows the matter type, as ``DynamicCollector.Inlet``
        selects its model, rather than the source class: mixed-phase
        (``PharmaPy.MixedPhases``) matter, such as a ``SlurryStream`` from an
        ``MSMPR`` or a continuous solids ``Mixer``, gets the crystallizer
        names, and a liquid source the liquid-mixer names.

        ``NameAnalyzer`` then pairs them with the source's published names.
        An ``MSMPR`` publishes ``mass_conc`` [kg/m**3], passed through
        unchanged. A source publishing liquid mass fractions ``mass_frac``
        [-], such as the solids ``Mixer``, has them converted to liquid mass
        concentration [kg/m**3] by the attached liquid's ``frac_to_conc``:
        ``w_i / sum_j(w_j / rho_j)``, the fractions times the ideal
        mass-basis mixing density of the database pure-component liquid
        densities ``rho_liq`` [kg/m**3], which are temperature independent.
        Fields the source does not publish, such as the solids Mixer's
        ``vol_flow`` [m**3/s], are read by the collector from the
        transferred stream itself.
        """
        names = self.destination_uo.names_states_in
        if self.Matter.__module__ == 'PharmaPy.MixedPhases':
            return names['crystallizer']
        return names['liquid_mixer']

    def ConvertUnits(self) -> None:
        """Convert upstream states using the destination's selected names.

        Notes
        -----
        Resolve a flow/non-flow name selector from the transferred matter
        before name analysis. Mixer inlet assignment happens after conversion,
        so an unconnected mixer still exposes both alternatives here. Stream
        quantities retain their per-second basis; batch amounts retain their
        inventory basis. Converted states are stored in ``Matter.y_inlet``.
        A ``DynamicCollector`` destination receives the input names of the
        model its inlet selects, from :meth:`_collector_input_names`.
        """
        mode_source = self.source_uo.oper_mode
        mode_dest = self.destination_uo.oper_mode

        flow_flag = (mode_source == 'Continuous' and mode_dest != 'Batch')
        btf_flag = self.source_uo.__class__.__name__ == 'BatchToFlowConnector'

        if flow_flag or btf_flag:
            names_states_in = self.destination_uo.names_states_in
            if isinstance(names_states_in, dict) and 'flow' in names_states_in:
                selector = 'flow' if hasattr(self.Matter, 'mass_flow') else 'non_flow'
                names_states_in = names_states_in[selector]

        if flow_flag:
            states_up = self.source_uo.names_states_out

            class_destination = self.destination_uo.__class__.__name__
            if class_destination == 'DynamicCollector':
                states_down = self._collector_input_names()
            else:
                states_down = names_states_in
            
            if 'mu_n' in states_up:
                num_distr = len(self.Matter.moments)
            elif 'distrib' in states_up:
                num_distr = len(self.Matter.distrib)
            else:
                num_distr = 0
                
            name_analyzer = NameAnalyzer(
                states_up, states_down, self.num_species,
                num_distr)

            # Convert units and pass states to self.Matter
            converted_states = name_analyzer.convertUnits(self.Matter)
            self.Matter.y_inlet = converted_states

        elif btf_flag:
            states_up = self.source_uo.names_states_out

            class_destination = self.destination_uo.__class__.__name__
            if class_destination == 'DynamicCollector':
                states_down = self._collector_input_names()
            else:
                states_down = names_states_in

            name_analyzer = NameAnalyzer(
                states_up, states_down, self.num_species,
                len(getattr(self.Matter, 'distrib', []))
                )

            # Convert units and pass states to self.Matter
            converted_states = name_analyzer.convertUnits(self.Matter)
            self.Matter.y_inlet = converted_states

    def PassPhases(self):

        class_destination = self.destination_uo.__class__.__name__
        mode_dest = self.destination_uo.oper_mode
        transfered_matter = copy.deepcopy(self.Matter)

        transfered_matter.transferred_from_uo = True

        if class_destination == 'Mixer':
            self.destination_uo.Inlets = transfered_matter

        elif mode_dest == 'Batch':
            self.destination_uo.Phases = transfered_matter

        elif mode_dest == 'Semibatch':
            if class_destination == 'DynamicCollector':
                self.destination_uo.Inlet = transfered_matter
                self.destination_uo.material_from_upstream = True

                if self.source_uo.__module__ == 'PharmaPy.Crystallizers':
                    self.destination_uo.KinCryst = self.source_uo.Kinetics
                    self.destination_uo.kwargs_cryst = {
                        'target_ind': self.source_uo.target_ind,
                        'target_comp': self.source_uo.target_comp,
                        'scale': self.source_uo.scale}

            elif self.destination_uo.Phases is None:
                self.destination_uo.Phases = transfered_matter
                self.destination_uo.material_from_upstream = True

        elif mode_dest == 'Continuous':  # Continuous
            # Transfering from batch to continuous (how to approach this?)
            if self.source_uo.oper_mode != 'Continuous':
                pass
                # TODO: big TODO. We need to define how Batch/Semibatch
                # followed by continuous will be handled. The most practical
                # approach would be to solve thhe the downstream continuous
                # section for a period of time such as the material from the
                # last discontinuous UO is depleted, as stated in the paper.
                # Reference date: (2022/06/28)

            if class_destination == 'DynamicExtractor':
                self.destination_uo.Inlet = {'feed': transfered_matter}
            else:
                self.destination_uo.Inlet = transfered_matter
