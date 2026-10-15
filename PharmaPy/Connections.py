# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:36:35 2020

@author: dcasasor
"""

from collections import deque
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from PharmaPy.Errors import PharmaPySpecificationError, PharmaPyValueError
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


# Marks a held object without a ``name`` before a failed batch assignment.
_UNNAMED = object()

# Operating modes by which PassPhases routes matter into a destination other
# than a Mixer, which derives its own mode from the basis of its inlets.
DESTINATION_MODES = ('Batch', 'Semibatch', 'Continuous')

# Attributes every flowsheet source must declare: FeedConnection reads its
# outlet, published states and operating flags (issue #445).
SOURCE_CONTRACT = ('Outlet', 'outputs', 'is_continuous', 'oper_mode')


def _unit_label(unit: object, name: Union[str, None]) -> str:
    """Return a unit's flowsheet name and class for error messages.

    Parameters
    ----------
    unit : object
        Unit operation instance.
    name : str or None
        Flowsheet name of the unit, if known.

    Returns
    -------
    str
        ``"'NAME' (Class)"``, or ``"Class"`` without a name.
    """
    class_name = type(unit).__name__
    if name is None:
        return class_name
    return f"'{name}' ({class_name})"


class Connection:
    def __init__(self, source_uo, destination_uo, replaced_matter=None,
                 replaced_phases=(), source_name=None, destination_name=None):
        """Connect a solved source unit's outlet to a destination unit.

        Parameters
        ----------
        source_uo : object
            Solved unit operation whose ``Outlet`` is transferred.
        destination_uo : object
            Unit operation that receives a deep copy of the outlet.
        replaced_matter : object, optional
            Matter that an earlier transfer on the same source-destination
            edge handed to ``destination_uo``, that is, the earlier
            connection's ``transferred_matter``. If the destination still
            holds it, :meth:`PassPhases` replaces it with the new copy
            instead of adding the new copy to it (issue #441). The default,
            None, keeps every input the destination holds.
        replaced_phases : sequence, optional
            Phase objects that ``replaced_matter`` carried when it was
            transferred, that is, the earlier connection's
            ``transferred_phases``. A batch destination drops them together
            with ``replaced_matter``. They are given explicitly because a
            destination may later add its own phases to the transferred
            cake's live ``Phases`` list (``Drying`` keeps that list as its
            own). Empty by default.
        source_name, destination_name : str, optional
            Flowsheet names of the two units, used only to name them in
            error messages; by default the messages name their classes.
        """
        self.source_uo = source_uo
        self.destination_uo = destination_uo
        self.replaced_matter = replaced_matter
        self.replaced_phases = tuple(replaced_phases)
        self.source_name = source_name
        self.destination_name = destination_name
        # Deep copy handed to the destination by PassPhases, and the phases
        # it carried at that time; SimulationExec records both to replace
        # the copy on a later transfer and to verify raw-material use.
        self.transferred_matter = None
        self.transferred_phases = ()

    def transfer_data(self) -> None:
        """Check the connection, then hand the source outlet downstream.

        Raises
        ------
        PharmaPySpecificationError
            If either unit lacks the connection contract; see
            :meth:`_check_contract`.
        PharmaPyValueError
            If the destination cannot take the source's amount basis; see
            :meth:`_check_route`.

        Notes
        -----
        Both checks run before :meth:`FeedConnection`, so a rejected
        connection leaves the source outlet and the destination unchanged.
        Then :meth:`FeedConnection` attaches the source profile,
        :meth:`ConvertUnits` converts it to the destination's input names
        and :meth:`PassPhases` installs a copy in the destination.
        """
        self._check_contract()
        self._check_route()
        self.FeedConnection()
        self.ConvertUnits()
        self.PassPhases()

    def _conversion_flags(self) -> Tuple[bool, bool]:
        """Return which name conversion :meth:`ConvertUnits` applies.

        Returns
        -------
        flow_flag : bool
            True if the source's ``oper_mode`` is ``'Continuous'`` and the
            destination's is not ``'Batch'``: stream states are converted
            to the destination's ``names_states_in``.
        btf_flag : bool
            True if the source is a ``BatchToFlowConnector``, whose
            discharge stream is converted likewise.
        """
        flow_flag = (self.source_uo.oper_mode == 'Continuous'
                     and self.destination_uo.oper_mode != 'Batch')
        btf_flag = (self.source_uo.__class__.__name__
                    == 'BatchToFlowConnector')
        return flow_flag, btf_flag

    def _check_contract(self) -> None:
        """Reject units that lack the attributes the connection reads.

        Raises
        ------
        PharmaPySpecificationError
            If the source lacks ``Outlet``, ``outputs``, ``result.time``,
            ``is_continuous`` or ``oper_mode``; if the destination lacks
            ``oper_mode``; if the states are converted (see
            :meth:`_conversion_flags`) and the source lacks
            ``names_states_out`` or the destination ``names_states_in``.
            The message names each unit, its class and the missing
            attributes. Also if a destination other than a ``Mixer``
            declares an ``oper_mode`` outside ``DESTINATION_MODES``, such
            as the lowercase ``'batch'`` default of
            ``PharmaPy.MetaModeler.MetaModelingClass``, for which
            :meth:`PassPhases` would install nothing; that message names
            the unit, its class and the mode.

        Notes
        -----
        ``PharmaPySpecificationError`` subclasses ``AttributeError``, which
        such connections raised before this check (issue #445), and the
        defect is in the unit's specification for flowsheet use, not in
        the values it holds. A unit whose outlet or published time profile
        is missing, such as a flash or extractor, can still be the last
        unit of a flowsheet, but not a source.

        The check covers what real units can lack. Every PharmaPy unit
        that declares a supported mode also declares the ``Inlet`` or
        ``Phases`` setter of that mode (a ``Mixer`` its ``Inlets``), and
        the only unit with a dictionary ``Outlet``, ``DynamicExtractor``,
        declares its ``default_output``, so neither is checked.
        """
        source, destination = self.source_uo, self.destination_uo
        is_mixer = destination.__class__.__name__ == 'Mixer'

        missing_source = [name for name in SOURCE_CONTRACT
                          if not hasattr(source, name)]
        if not hasattr(getattr(source, 'result', None), 'time'):
            missing_source.append('result.time')

        missing_destination = [] if hasattr(destination, 'oper_mode') else [
            'oper_mode']
        mode = getattr(destination, 'oper_mode', None)
        if not is_mixer and not missing_destination and (
                mode not in DESTINATION_MODES):
            raise PharmaPySpecificationError(
                f'{_unit_label(destination, self.destination_name)} '
                f'declares oper_mode {mode!r}; a flowsheet destination '
                'other than a Mixer must declare one of '
                f'{list(DESTINATION_MODES)} so that Connection can install '
                'the transferred matter.')

        if 'oper_mode' not in missing_source + missing_destination:
            flow_flag, btf_flag = self._conversion_flags()
            if flow_flag or btf_flag:
                if not hasattr(source, 'names_states_out'):
                    missing_source.append('names_states_out')
                if not hasattr(destination, 'names_states_in'):
                    missing_destination.append('names_states_in')

        problems = []
        if missing_source:
            problems.append(
                f'source {_unit_label(source, self.source_name)} lacks '
                f'{missing_source}, which a flowsheet source must publish')
        if missing_destination:
            problems.append(
                'destination '
                f'{_unit_label(destination, self.destination_name)} lacks '
                f'{missing_destination}, which a flowsheet destination must '
                'declare')
        if problems:
            raise PharmaPySpecificationError(
                'Cannot connect these units: ' + '; '.join(problems)
                + '. The unit class does not implement the connection '
                'contract for this role; use it standalone (assign its '
                'Inlet or Phases and call solve_unit) or connect a unit '
                'that implements it.')

    def _check_route(self) -> None:
        """Reject a source whose amount basis the destination cannot take.

        Raises
        ------
        PharmaPyValueError
            If a source that is not continuous (``is_continuous`` False)
            feeds a continuous destination (``oper_mode`` ``'Continuous'``)
            or a ``DynamicCollector``, or if a continuous source feeds a
            batch destination (``oper_mode`` ``'Batch'``, including a
            ``BatchToFlowConnector``). The message names both units, the
            basis mismatch and the remedy.

        Notes
        -----
        The route is keyed on the source's ``is_continuous``, not its
        ``oper_mode``: a ``BatchToFlowConnector`` has ``oper_mode`` 'Batch'
        but publishes a stream. A continuous source hands over a stream
        with a flow rate [kg/s]; any other source hands over its final
        holdup, an amount [kg] with no flow (see :meth:`FeedConnection`).
        A continuous destination and a ``DynamicCollector`` integrate a
        feed flow, and a batch destination is charged with a holdup, so
        neither can take the other basis: a holdup as ``Inlet`` failed
        later on a missing ``vol_flow`` or ``DynamicInlet``, and a stream
        as ``Phases`` was rejected by the setter, failed on a missing
        ``vol``, or was charged as an amount (issue #436). The remedies
        exist: a ``BatchToFlowConnector`` discharges a liquid holdup as a
        stream over a cycle time (solids are not supported, issue #423),
        and a ``DynamicCollector`` collects a stream for a duration into a
        holdup that can charge a batch unit. ``PharmaPyValueError`` is used,
        as for the fan-in check of ``SimulationExec``, because the
        flowsheet specification is inconsistent and a remedy exists.

        A ``Mixer`` destination is not checked: it adopts the basis of its
        inlets and rejects mixing inventories with flow rates itself.
        Semibatch destinations other than a ``DynamicCollector`` are not
        checked here.
        """
        destination = self.destination_uo
        class_destination = destination.__class__.__name__
        if class_destination == 'Mixer':
            return

        source_label = _unit_label(self.source_uo, self.source_name)
        destination_label = _unit_label(destination, self.destination_name)
        continuous_source = bool(self.source_uo.is_continuous)
        mode = destination.oper_mode

        if not continuous_source and (class_destination == 'DynamicCollector'
                                      or mode == 'Continuous'):
            raise PharmaPyValueError(
                f'Source {source_label} is not continuous: its outlet is a '
                'holdup, an amount [kg] without a flow rate, but '
                f'destination {destination_label} takes a feed stream '
                '[kg/s] as its Inlet. Insert a BatchToFlowConnector '
                'between them to discharge the holdup as a stream over a '
                'cycle time; it accepts a single liquid holdup only, '
                'solids are not supported yet (issue #423).')

        if continuous_source and mode == 'Batch':
            raise PharmaPyValueError(
                f'Source {source_label} is continuous: its outlet is a '
                'stream, a flow rate [kg/s], but destination '
                f'{destination_label} is a batch unit charged with a '
                'holdup, an amount [kg], as its Phases. Collect the stream '
                'for a duration in a DynamicCollector and connect the '
                'collector, whose holdup can charge the batch unit.')

    def FeedConnection(self) -> None:
        """Select the source outlet and attach its upstream profile.

        Notes
        -----
        ``self.Matter`` is the source's whole ``Outlet``, or
        ``Outlet[default_output]`` when the outlet is a dictionary. Its
        ``y_upstream`` is set to the source's ``outputs`` and its
        ``time_upstream`` to the source's result times [s] (continuous
        source) or final time [s] (batch source). The outlet is never
        split, so a source connected to several destinations would hand
        each of them the full flow [kg/s] or inventory [kg];
        ``SimulationExec.SolveFlowsheet`` therefore rejects such fan-out
        graphs before solving (issue #437).
        """
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
        flow_flag, btf_flag = self._conversion_flags()

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

    def PassPhases(self) -> None:
        """Hand a deep copy of the source outlet to the destination unit.

        Raises
        ------
        Exception
            Any exception raised by the destination's ``Inlets``, ``Inlet``
            or ``Phases`` setter propagates, for example
            ``NotImplementedError``/``TypeError``/``ValueError`` when a
            ``BatchToFlowConnector`` rejects anything other than one liquid
            batch holdup (issue #423), ``ValueError`` when a ``Mixer``
            rejects mixing batch inventories [kg] with flow rates [kg/s],
            or replacing its connected copy with one on the other basis,
            or ``RuntimeError`` when a reactor, crystallizer or filter
            ``Phases`` setter receives a stream or a non-phase object.

        Notes
        -----
        The transferred matter is ``copy.deepcopy(self.Matter)``, so the
        source outlet is never shared, and it is marked
        ``transferred_from_uo = True`` (raw-material reporting excludes it).
        It is kept as ``self.transferred_matter`` so ``SimulationExec`` can
        verify that the destination actually uses it.
        Dispatch by destination:

        * ``Mixer``: appended to ``Inlets``.
        * Batch units, including ``BatchToFlowConnector``: assigned to
          ``Phases``.
        * ``DynamicCollector``: assigned to ``Inlet``, which selects the
          collector model from the matter type, with
          ``material_from_upstream = True``. Only a ``PharmaPy.Crystallizers``
          source also sets ``KinCryst`` to its ``Kinetics`` and
          ``kwargs_cryst`` to its ``target_ind``, ``target_comp`` and
          ``scale``; for other sources caller-set values are kept.
        * Other semibatch units: assigned to ``Phases`` only if none are
          set, with ``material_from_upstream = True``; a transfer to an
          already-charged unit is ignored (issue #438).
        * Continuous units: assigned to ``Inlet`` (``{'feed': matter}`` for
          a ``DynamicExtractor``).

        The matter keeps the source's amount basis, so :meth:`_check_route`
        has already rejected the routes that would change it (issue #436):
        the holdup [kg] of a source that is not continuous reaches a
        continuous unit or a ``DynamicCollector`` only through a
        ``BatchToFlowConnector`` (liquid holdups only), and the stream
        [kg/s] of a continuous source reaches a batch unit only after a
        ``DynamicCollector`` has collected it into a holdup.

        A transfer replaces the destination's copy of ``replaced_matter``
        rather than adding to it, keyed by object identity, so repeating a
        transfer on the same edge gives the same inputs (issue #441):

        * ``Mixer``: the copy is replaced in its position in ``Inlets``,
          and the Mixer's other inlets, such as raw feeds, are kept.
        * Batch units: the copy, and the phases a transferred ``Cake`` or
          ``Slurry`` carried (``replaced_phases``), are removed from a
          list-valued ``Phases`` before the assignment, so a ``BatchCryst``
          or ``Drying`` unit, whose setter adds single phases or cake
          phases to its holdup, does not stack the new transfer on the old
          one. Phases set independently, such as a crystallizer's seed or a
          dryer's gas, are kept. If the assignment raises, the destination's
          attributes, phase list and phase names are restored.
        * ``DynamicExtractor``: ``{'feed': matter}`` replaces the held
          ``'feed'``, because its ``Inlet`` setter lets newly assigned keys
          win (issue #449); its ``'solvent'`` is kept.

        Without ``replaced_matter``, or when the destination no longer
        holds it, the transfer is installed as described above.

        Only a ``Mixer`` combines several transfers into one mass- and
        energy-balanced mixture. A transfer from a further source to any
        other destination replaces its ``Inlet`` or ``Phases``, is ignored
        (a charged semibatch unit), or, for a ``BatchCryst`` or ``Drying``
        unit, is stacked with the earlier phases into a holdup list that is
        not a mixture of the feeds (a ``Slurry`` replaces a
        ``BatchCryst``'s phases instead).
        ``SimulationExec.SolveFlowsheet`` therefore rejects several
        predecessors of a non-``Mixer`` unit before solving (issue #444).
        """
        class_destination = self.destination_uo.__class__.__name__
        mode_dest = self.destination_uo.oper_mode
        transfered_matter = copy.deepcopy(self.Matter)

        transfered_matter.transferred_from_uo = True
        self.transferred_matter = transfered_matter
        # Snapshot before installation: a destination may extend the list.
        self.transferred_phases = tuple(
            getattr(transfered_matter, 'Phases', None) or ())

        replaced = self.replaced_matter
        if class_destination == 'Mixer':
            if replaced is None or not self.destination_uo._replace_inlet(
                    replaced, transfered_matter):
                self.destination_uo.Inlets = transfered_matter

        elif mode_dest == 'Batch':
            self._assign_batch_phases(transfered_matter)

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

        elif mode_dest == 'Continuous':
            if class_destination == 'DynamicExtractor':
                self.destination_uo.Inlet = {'feed': transfered_matter}
            else:
                self.destination_uo.Inlet = transfered_matter

    def _assign_batch_phases(self, matter) -> None:
        """Assign transferred matter to a batch destination's ``Phases``.

        Parameters
        ----------
        matter : object
            Deep copy of the source outlet: a phase, or a ``Cake`` or
            ``Slurry`` holding phases, with inventories [kg].

        Raises
        ------
        Exception
            Any exception raised by the destination's ``Phases`` setter
            propagates after the destination is restored (see Notes).

        Notes
        -----
        ``replaced_matter`` and ``replaced_phases`` are removed by identity
        from the destination's ``Phases`` list in place, so setters that add
        to the held list (``BatchCryst``, ``Drying``) do not keep the
        earlier transfer. Setters that replace their phases are unaffected.
        A destination whose ``Phases`` is not a list, or a transfer without
        ``replaced_matter``, is assigned directly.

        A setter can fail after it has changed the destination, as a
        ``Drying`` unit does when the new cake lacks one of its
        ``supercrit_names`` species after re-classifying its phases and
        resizing its cake grid. The destination's attribute bindings (a
        shallow copy of its ``__dict__``), the contents of its phase list
        and the ``name`` of each held phase are therefore restored before
        the exception propagates. Objects the setter changed in place
        other than these are not restored.
        """
        destination = self.destination_uo
        held = destination.Phases
        replaced = self.replaced_matter
        if replaced is None or not isinstance(held, list):
            destination.Phases = matter
            return

        stale = (replaced,) + self.replaced_phases
        attributes = dict(vars(destination))
        original = list(held)
        names = [(phase, getattr(phase, 'name', _UNNAMED))
                 for phase in original]
        held[:] = [phase for phase in held
                   if not any(phase is old for old in stale)]
        try:
            destination.Phases = matter
        except Exception:
            vars(destination).clear()
            vars(destination).update(attributes)
            held[:] = original
            for phase, name in names:
                if name is not _UNNAMED:
                    phase.name = name
            raise
