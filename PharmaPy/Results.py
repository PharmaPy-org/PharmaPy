# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:38:23 2022

@author: dcasasor
"""

from collections import Counter
from typing import List, Mapping, Sequence

import numpy as np
import pandas as pd


def get_name_object(obj):
    typ = obj.__class__.__name__
    identif = repr(obj).split(' ')[-1][:-1]

    return '.'.join([typ, identif])


def get_stream_info(obj, fields):
    if not isinstance(obj, (tuple, list)):
        obj = [obj]

    out = {}
    for ind, phase in enumerate(obj):
        stream_info = {}
        for field in fields:
            stream_info[field] = getattr(phase, field, None)

        phase_name = get_name_object(phase)
        out[phase_name] = stream_info

    return out


def flatten_dict_fields(di, index=None):
    target_keys = []

    new_fields = {}
    for key in di:
        if isinstance(di[key], (tuple, list, np.ndarray)):
            val = di[key]
            target_keys.append(key)

            if index is None:
                suff = range(len(val))
            elif isinstance(index, (list, tuple)):
                suff = index
            else:
                suff = index[key]

            for ind, num in enumerate(val):
                new_fields['%s_%s' % (key, suff[ind])] = num

    for key in target_keys:
        di.pop(key)

    out = {**di, **new_fields}
    return out


def get_di_multiindex(di):

    out = {(i, j, k): di[i][j][k]
           for i in di.keys()
           for j in di[i].keys()
           for k in di[i][j].keys()}

    return out


def pprint(di, name_items, fields, str_out=True):
    """
    Create a table showing items with their respective fiels as columns

    Parameters
    ----------
    di : dict
        dictionary structured as:

            {'item_1': {'field_1':..., 'field_2':...},
             'item_2': {'field_1':..., 'field_2':...},
             ...}

    described_name : str
        name of described variable.
    fields : list of str
        name of fields to be taken from nested dictionaries.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """

    out = []

    form_header = []
    form_vals = []
    lens_header = []

    header = [name_items] + list(fields.keys())

    items = list(di.keys())

    # ---------- Lenghts
    # Lenght of first column
    max_lens = {name_items: max([len(name) for name in items])}

    # Lenght of remaining columns
    for field in fields:
        field_vals = [di[name].get(field, '') for name in items]

        for ind, val in enumerate(field_vals):
            if isinstance(val, list):
                if all([type(a) == str for a in val]):
                    field_vals[ind] = ', '.join(val)
                else:
                    field_vals[ind] = '%i, ..., %i' % (val[0], val[-1])

        len_vals = [len(repr(val)) for val in field_vals]
        max_lens[field] = max(len_vals)

    # All fields
    all_fields = {name_items: 's', **fields}
    for name in all_fields:
        le = max(len(name), max_lens[name]) + 2

        form_header.append("{:<%i}" % le)
        form_vals.append("{:<%i%s}" % (le, all_fields[name]))

        lens_header.append(le)

    form_header = ' '.join(form_header)
    form_vals = ' '.join(form_vals)

    len_headers = sum(lens_header)

    lines = '-' * len_headers
    out.append(lines)
    out.append(form_header.format(*header))
    out.append(lines)

    for name in di:
        field_vals = [di[name].get(field, '') for field in fields]

        for ind, val in enumerate(field_vals):
            if isinstance(val, list):
                if all([type(a) == str for a in val]):
                    field_vals[ind] = ', '.join(val)
                else:
                    field_vals[ind] = '%i, ..., %i' % (val[0], val[-1])

        item = form_vals.format(*([name] + field_vals))

        out.append(item)

    out.append(lines)

    if str_out:
        out = '\n'.join(out)

    return out


class DynamicResult:
    def __init__(self, di_states, di_fstates=None, **results):
        self.__dict__.update(**results)
        self.di_states = di_states
        self.di_fstates = di_fstates

    def __repr__(self):
        headers = {'dim': '', 'units': 's', 'index': 's'}

        str_states = pprint(self.di_states, 'states', headers)

        state_example = list(self.di_states.keys())[0]

        header = 'PharmaPy result object'
        lines = '-' * (len(header) + 2)

        header = '\n'.join([lines, header, lines + '\n\n'])

        explain = 'Fields shown in the tables below can be accessed as ' \
            'result.<field>, e.g. result.%s \n\n' % state_example

        top_text = header + explain

        if self.di_fstates is not None and len(self.di_fstates) > 0:
            head = {'dim': '', 'units': 's', 'index': 's'}
            str_fstates = pprint(self.di_fstates, 'f(states)', head)

            out_str = str_states + '\n\n' + str_fstates

        else:
            out_str = str_states

        out_str = top_text + out_str
        out_str += '\n\nTime vector can be accessed as result.time\n'

        return out_str


def _flowsheet_diagram_lines(graph: Mapping[str, Sequence[str]],
                             execution_names: Sequence[str]) -> List[str]:
    """Render the actual connections of a flowsheet graph as text lines.

    Parameters
    ----------
    graph : mapping of str to sequence of str
        Directed flowsheet adjacency in declaration order. Successors may be
        any ordered sequence, such as a list, tuple or NumPy array; units
        that appear only as successors have no outgoing connections.
    execution_names : sequence of str
        Unit names in execution order, as returned by
        ``PharmaPy.Connections.topological_bfs``: every unit for an acyclic
        graph, only the units scheduled before a cycle otherwise.

    Returns
    -------
    list of str
        A single ``'A --> B --> C'`` line, in execution order, when the graph
        is exactly one connected directed path (including a single unit).
        Otherwise one ``'source --> destination'`` line per graph edge,
        including self-loops and edges of unscheduled units, in ``graph``
        declaration order, with each unconnected unit's bare name at its
        declaration position. Empty when there are no units.

    Notes
    -----
    Edges and degrees are taken from ``graph``, not from the residual
    counters of ``topological_bfs``. The linear form requires that the
    execution order covers every graph node (keys and successors) and that
    its consecutive pairs are exactly the graph's edges, so it never invents
    a connection or hides a unit left unscheduled by a recycle.
    """
    edges = [(source, destination)
             for source, successors in graph.items()
             for destination in successors]
    in_degree = Counter(destination for _, destination in edges)  # [-]
    out_degree = Counter(source for source, _ in edges)  # [-]
    nodes = set(graph).union(in_degree)

    names = list(execution_names)
    is_single_path = (len(names) > 0 and set(names) == nodes
                      and len(edges) == len(names) - 1
                      and set(zip(names, names[1:])) == set(edges))

    if is_single_path:
        return [' --> '.join(names)]

    lines = []
    for source, successors in graph.items():
        lines.extend('{} --> {}'.format(source, destination)
                     for destination in successors)
        if out_degree[source] == 0 and in_degree[source] == 0:
            lines.append(source)

    return lines


class SimulationResult:
    def __init__(self, sim):
        self.sim = sim

        # Create UO summary
        di_uos = {}

        names_uos = sim.execution_names

        headers = {'Diff eqns': 'd', 'Alg eqns': 'd',
                   'Model type': 's', 'PharmaPy type': 's'}

        for name in names_uos:
            uo = getattr(sim, name)
            states_di = getattr(uo, 'states_di', None)

            num_discr = 1

            if hasattr(uo, 'num_discr'):
                num_discr = uo.num_discr
            elif hasattr(uo, 'number_nodes'):
                num_discr = uo.number_nodes

            if states_di is not None:
                num_diff = []
                num_alg = []
                for var, di in states_di.items():
                    num = di.get('index', 1)

                    if isinstance(num, (list, tuple)):
                        num = len(num)

                    num *= num_discr

                    if 'type' in di:
                        if di['type'] == 'diff':
                            num_diff.append(num)
                        elif di['type'] == 'alg':
                            num_alg.append(num)

                if sum(num_diff) == 0:
                    model_type = 'ALG'
                elif sum(num_diff) > 0 and sum(num_alg) > 0:
                    model_type = 'DAE'
                else:
                    model_type = 'ODE'

                di_uos[name] = {}

                pharmapy_type = getattr(sim, name).__class__.__name__

                di_uos[name]['Diff eqns'] = sum(num_diff)
                di_uos[name]['Alg eqns'] = sum(num_alg)
                di_uos[name]['Model type'] = model_type
                di_uos[name]['PharmaPy type'] = pharmapy_type

        out_uos = pprint(di_uos, 'Unit operation', headers, str_out=False)

        self.out_uos = out_uos

    def GetStreamTable(self, basis: str = 'mass') -> pd.DataFrame:
        """Return raw-material and outlet records in flowsheet execution order.

        Parameters
        ----------
        basis : {'mass', 'mole'}, optional
            Amount and composition basis; defaults to mass.

        Returns
        -------
        pandas.DataFrame
            Rows indexed by unit operation, source (inlet, initial holdup,
            or outlet), and phase/stream identifier. Columns include
            temperature [K], pressure [Pa], volume [m**3], volumetric flow
            [m**3/s], and species fractions [-], where applicable. Mass
            amounts and flows are [kg] and [kg/s]; molar amounts and flows
            are [mol] and [mol/s]. Inapplicable fields are NaN.

        Raises
        ------
        ValueError
            If ``basis`` is not ``'mass'`` or ``'mole'``, or (forwarded from
            ``SimulationExec.GetRawMaterials``) if a raw inlet's dynamic
            controls cannot be accounted.

        Warns
        -----
        RuntimeWarning
            Forwarded from ``SimulationExec.get_raw_inlets``, once per
            instantaneous unit whose raw amounts are NaN.

        Notes
        -----
        Continuous raw inlet amounts integrate over the receiving unit's
        fed duration. An instantaneous unit (a static continuous ``Mixer``)
        is charged for as long as its single, verified downstream consumer
        consumed its latest transfer; otherwise its amounts are NaN. See
        ``SimulationExec.get_raw_inlets`` for the full list of NaN reasons.
        Raw rows come from ``GetRawMaterials(totals=False)``: inlets with a
        ``DynamicInlet`` report the integrated feed the unit consumed, with
        amounts, volume, fractions averaged with the basis flow, the consumed
        temperature (mass-flow-weighted when it varies), static pressure,
        and NaN flow-rate columns.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be either 'mass' or 'mole'")

        uo_dict = self.sim.uos_instances

        base = ['temp', 'pres']

        if basis == 'mass':
            fields_phase = base + ['mass', 'vol', 'mass_frac']
            fields_stream = base + ['mass_flow', 'vol_flow', 'mass_frac']

        elif basis == 'mole':
            fields_phase = base + ['moles', 'vol', 'mole_frac']
            fields_stream = base + ['mole_flow', 'vol_flow', 'mole_frac']

        info = {}
        for ind, name in enumerate(uo_dict):
            matter_obj = uo_dict[name].Outlet

            if not isinstance(matter_obj, dict):
                matter_obj = {'Outlet': matter_obj}
            else:
                matter_obj = {'Outlet_%s' % key: val
                              for key, val in matter_obj.items()}

            info[name] = {}

            for out_name, matter in matter_obj.items():
                if 'Stream' in matter.__class__.__name__:
                    fields = fields_stream
                else:
                    fields = fields_phase

                if matter.__module__ == 'PharmaPy.MixedPhases':
                    matter = matter.Phases

                entries = get_stream_info(matter, fields)
                entries = {key: flatten_dict_fields(val, self.sim.NamesSpecies)
                           for key, val in entries.items()}

                info[name][out_name] = entries

        di_multiindex = get_di_multiindex(info)
        mux = pd.MultiIndex.from_tuples(di_multiindex.keys())

        stream_table = pd.DataFrame(list(di_multiindex.values()), index=mux)

        raw_materials = self.sim.GetRawMaterials(basis=basis, totals=False)
        stream_table = pd.concat((raw_materials, stream_table), axis=0)

        grouped = stream_table.groupby(level=0)

        dfs = []

        for key in uo_dict:
            dfs.append(grouped.get_group(key))

        stream_table = pd.concat(dfs, axis=0)

        return stream_table

    def __repr__(self) -> str:
        """Summarize the flowsheet connections and unit-operation models.

        Returns
        -------
        str
            Welcome banner, the flowsheet structure rendered by
            ``_flowsheet_diagram_lines`` from the simulation graph, and the
            per-unit table of equation counts, model types and classes.
        """
        # Welcome message
        welcome = 'Welcome to PharmaPy'
        len_header = len(welcome) + 2
        lines = '-' * len_header

        names_uos = self.sim.execution_names
        out = [lines, welcome, lines + '\n']

        # Flowsheet ASCII diagram of the actual graph edges
        if names_uos is not None:
            flow_diagram = '\n'.join(
                _flowsheet_diagram_lines(self.sim.graph, names_uos))

            out += ['Flowsheet structure:', flow_diagram + '\n']

        # Include UOs table
        out_str = '\n'.join(out + self.out_uos)

        return out_str
