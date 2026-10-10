# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:02:01 2022

@author: dcasasor
"""

import matplotlib.pyplot as plt
import numpy as np

from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Commons import retrieve_pde_result


special = ('alpha', 'beta', 'gamma', 'phi', 'rho', 'epsilon', 'sigma', 'mu',
           'nu', 'psi', 'pi', '#')


def latexify_name(name, units=False):
    parts = name.split('/')

    out = []
    count = 0
    for part in parts:
        sep = None
        if '**' in part:
            segm = part.split('**')
            sep = '^'
        elif '_' in part:
            segm = part.split('_')
            sep = '_'
        else:
            segm = [part]

        for ind, s in enumerate(segm):
            if s in special:
                segm[ind] = '\\' + s

        if sep is None:
            if count > 0:
                part = part + '^{-1}'
            else:
                part = segm[0]
        else:
            inv = ''
            if count > 0:
                inv = '-'

            part = segm[0] + sep + '{' + inv + segm[1] + '}'

        out.append(part)
        count += 1

    if len(out) > 1:
        out = ' \ '.join(out)
    else:
        out = out[0]

    if units:
        out = '$\mathregular{' + out + '}$'
    else:
        out = '$' + out + '$'

    return out


def format_unit_label(units):
    """Return a display label for unit metadata.

    Parameters
    ----------
    units : str or None
        Unit metadata. Bracketed values such as ``[K]`` and ``[-]`` follow the
        repository documentation convention.

    Returns
    -------
    str
        LaTeX-formatted unit label without metadata brackets. Dimensionless
        metadata ``[-]`` and empty values return an empty string.
    """
    if units is None:
        return ''

    units = str(units).strip()
    if units == '':
        return ''

    if units.startswith('[') and units.endswith(']'):
        units = units[1:-1].strip()

    if units in ('', '-'):
        return ''

    return latexify_name(units, units=True)


def color_axis(ax, color):
    ax.spines['right'].set_color(color)
    ax.tick_params(axis='y', colors=color, which='both')
    ax.yaxis.label.set_color(color)


def get_indexes(names, picks):
    names = [a for a in names]
    out = []

    lower_names = [str(a).lower() if isinstance(a, str) else a for a in names]

    for pick in picks:
        if isinstance(pick, str):
            low_pick = pick.lower()
            if low_pick in lower_names:
                out.append(lower_names.index(low_pick))
            else:
                mess = "Name '%s' not in the set of compound names listed in the pure-component json file" % low_pick
                raise PharmaPyValueError(mess)

        elif isinstance(pick, (int, np.int32, np.int64)):
            out.append(pick)

    return out


def get_state_data(uo, *state_names):

    time = uo.timeProf
    di = {}
    for name in state_names:
        idx = None
        if isinstance(name, (tuple, list, range)):
            state, idx = name
        else:
            state = name

        y = getattr(uo, state + 'Prof')
        if idx is not None:
            y = y[:, idx]

        di[state] = y

    return time, di


def get_state_names(state_list):
    out = []
    for state in state_list:
        if isinstance(state, (list, tuple)):
            state = state[0]
        out.append(state)

    return out


def get_state_distrib(result, *state_names, **kwargs_retrieve):

    states = get_state_names(state_names)
    di = retrieve_pde_result(result, states=states, **kwargs_retrieve)
    out = {}
    for name in state_names:
        idx = None
        if isinstance(name, (tuple, list, range)):
            state, idx = name
            indexes = result.di_states[state]['index']
            idx = [indexes[i]
                   if isinstance(i, (int, np.int32, np.int64)) else i
                   for i in idx]
        else:
            state = name

        y = di[state]
        if idx is None:
            if isinstance(y, dict):
                y = list(y.values())
        else:
            y = [y[i] for i in idx]

        out[state] = y

    return out


def _state_metadata(di_states, di_fstates) -> dict:
    """Merge state and function-of-state metadata, treating None as empty.

    Parameters
    ----------
    di_states : mapping or None
        State metadata (``units``, ``dim``, optional ``index``) keyed by
        state name.
    di_fstates : mapping or None
        Function-of-state metadata on the same layout. ``DynamicResult``
        and units without such quantities leave it None.

    Returns
    -------
    dict
        Combined metadata; a function-of-state entry overrides a state entry
        of the same name, as the former ``|`` merge did.
    """
    return {**(di_states or {}), **(di_fstates or {})}


def _with_time_axis(values, single_sample: bool, index=None):
    """Give a single-sample state its leading time axis when it lacks one.

    Parameters
    ----------
    values : array-like
        Published state values in their own units.
    single_sample : bool
        Whether the result's time has exactly one sample.
    index : sequence, optional
        The state's ``index`` metadata (for example species names).

    Returns
    -------
    numpy.ndarray or object
        For a single-sample result, a 0-d value becomes shape (1,) and a
        1-D indexed vector of length ``len(index)`` becomes (1, len(index)).
        Anything else, including every multi-sample result and every state
        that already has a time axis, is returned unchanged.
    """
    if not single_sample:
        return values
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1)
    if index is not None and array.ndim == 1 and len(array) == len(index):
        return array.reshape(1, -1)
    return values


def get_states_result(result, *state_names):
    """Return the time axis and selected states of a unit result.

    Parameters
    ----------
    result : DynamicResult
        Unit result exposing ``time`` [s], ``di_states``, optional
        ``di_fstates`` (None is treated as no function-of-state metadata)
        and one attribute per state in its own units.
    *state_names : str or tuple
        A state name, or a ``(name, picks)`` pair selecting entries of an
        indexed state by index name (case-insensitive) or position.

    Returns
    -------
    time : numpy.ndarray
        ``result.time`` [s], unchanged.
    out : dict
        State name mapped to its values in the result's units; picked
        entries of an indexed state keep the time axis first, with columns
        in pick order.

    Raises
    ------
    PharmaPyValueError
        If a picked name is not in the state's ``index``.

    Notes
    -----
    Instantaneous units, such as a batch ``Mixer``, may publish a single
    time sample with states that lack the leading time axis. When
    ``result.time`` has one sample, a 0-d state is returned with shape (1,)
    and a 1-D indexed state whose length equals its ``index`` with shape
    (1, len(index)), so it is handled like a multi-sample profile. States
    that already have a time axis, and all states of multi-sample results,
    are returned as published.
    """
    time = result.time
    states_fstates = _state_metadata(result.di_states, result.di_fstates)
    single_sample = np.size(time) == 1

    out = {}
    for key in state_names:
        idx = None
        if isinstance(key, (list, tuple, range)):
            state, idx = key
            indexes = states_fstates[state]['index']
            idx = get_indexes(indexes, idx)
        else:
            state = key

        y = _with_time_axis(getattr(result, state), single_sample,
                            states_fstates.get(state, {}).get('index'))

        if idx is not None:
            y = y[:, idx]

        out[state] = y

    return time, out


def name_yaxes(ax, states_fstates, names, ylabels, legend):
    for ind, name in enumerate(names):
        axis = ax[ind]
        if ylabels is None:
            ylabel = names[ind]
        else:
            ylabel = latexify_name(ylabels[ind])

        units = states_fstates[name].get('units', '')
        unit_name = format_unit_label(units)
        if len(unit_name) > 0:
            ylabel = ylabel + ' (' + unit_name + ')'

        axis.set_ylabel(ylabel)


def set_legend(ax, states_fstates, names, state_names, legend):
    for ind, name in enumerate(names):
        index_y = states_fstates[name].get('index', False)
        if index_y and legend:
            if isinstance(state_names[ind], (tuple, list)):
                picks = state_names[ind][1]
                picks = get_indexes(index_y, picks)

                index_y = [index_y[i] for i in picks]

            ax[ind].legend(index_y, loc='best')


def plot_function(uo, state_names, axes=None, fig_map=None, ylabels=None,
                  include_units=True, **fig_kwargs):
    """Plot selected state profiles of a solved unit against time.

    Parameters
    ----------
    uo : object
        Solved unit exposing ``result`` and ``states_di``; ``fstates_di`` is
        optional, and a missing or None value means no function-of-state
        metadata.
    state_names : sequence of str or tuple
        States to plot, each a name or a ``(name, picks)`` pair; see
        :func:`get_states_result`.
    axes : matplotlib.axes.Axes or numpy.ndarray, optional
        Axes to draw on; a new figure is created when omitted.
    fig_map : sequence of int, optional
        Axis position for each state; defaults to one axis per state.
    ylabels : sequence of str, optional
        Axis labels replacing the state names; units from the metadata are
        appended in parentheses.
    include_units : bool, optional
        Retained for API compatibility; units are always appended when the
        metadata defines them.
    **fig_kwargs
        Keyword arguments for ``matplotlib.pyplot.subplots``.

    Returns
    -------
    tuple or axes
        ``(figure, axes)`` when a figure is created, otherwise the axes.
    """
    time, data = get_states_result(uo.result, *state_names)

    if fig_map is None:
        fig_map = range(len(data))

    if axes is None:
        fig, ax_orig = plt.subplots(**fig_kwargs)
    else:
        ax_orig = axes

    if isinstance(ax_orig, np.ndarray):
        axes = ax_orig.flatten()
    else:
        axes = (ax_orig, )

    count = 0
    linestyles = ('-', '--', '-.', ':')
    colors = plt.cm.tab10

    names = list(data.keys())
    states_and_fstates = _state_metadata(
        getattr(uo, 'states_di', None), getattr(uo, 'fstates_di', None))

    for ind, idx in enumerate(fig_map):
        name = names[ind]
        y = data[name]
        twin = False

        index_y = states_and_fstates[name].get('index', False)

        if isinstance(state_names[ind], (tuple, list, range)):
            y_ind = state_names[ind][1]
            y_ind = get_indexes(index_y, y_ind)

            index_y = [index_y[a] for a in y_ind]

        if len(axes[idx].lines) > 0:
            ax = axes[idx].twinx()
            count += len(axes[idx].lines)
            twin = True
        else:
            ax = axes[idx]

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for sp, row in enumerate(y.T):
            ax.plot(time, row, color=colors(count),
                    linestyle=linestyles[count % len(linestyles)])

            if twin:
                color_axis(ax, colors(count))

            count += 1

        if ylabels is None:
            ylabel = name
        else:
            ylabel = latexify_name(ylabels[ind])

        units = states_and_fstates[name].get('units', '')
        unit_name = format_unit_label(units)
        if len(unit_name) > 0:
            ylabel = ylabel + ' (' + unit_name + ')'

        if index_y:
            ax.legend(index_y, loc='best')

        ax.set_ylabel(ylabel)

        count = 0

    if len(axes) == 1:
        axes = axes[0]

    if 'fig' in locals():
        return fig, ax_orig
    else:
        return ax_orig


def plot_distrib(uo, state_names, x_name, axes=None, times=None, x_vals=None,
                 cm_names=None, ylabels=None, legend=True, **fig_kwargs):
    """Plot distributed states along their internal coordinate or time.

    Parameters
    ----------
    uo : object
        Solved unit exposing ``result`` and ``states_di``; ``fstates_di`` is
        optional, and a missing or None value means no function-of-state
        metadata.
    state_names : sequence of str or tuple
        Distributed states, each a name or a ``(name, picks)`` pair.
    x_name : str
        Result attribute holding the internal coordinate, for example the
        crystal size grid [um].
    axes : matplotlib.axes.Axes or numpy.ndarray, optional
        Axes to draw on; a new figure is created when omitted.
    times : sequence of float, optional
        Times [s] at which to plot profiles along ``x_name``.
    x_vals : float or sequence of float, optional
        Coordinates, in the units of ``x_name``, at which to plot time
        profiles; used when ``times`` is None.
    cm_names : str or sequence of str, optional
        Matplotlib colormap names, one per plotted state entry.
    ylabels : sequence of str, optional
        Axis labels replacing the state names.
    legend : bool, optional
        Whether to label indexed states.
    **fig_kwargs
        Keyword arguments for ``matplotlib.pyplot.subplots``.

    Returns
    -------
    tuple or axes
        ``(figure, axes)`` when a figure is created, otherwise the supplied
        ``axes`` object itself, as in :func:`plot_function`. With
        ``times``, ``x_name`` labels the x axis as text on the axes' own
        figure.

    Raises
    ------
    ValueError
        If both ``times`` and ``x_vals`` are None.
    """
    if times is None and x_vals is None:
        raise ValueError("Both 'times' and 'x_vals' arguments are None. "
                         "Please specify one of them")

    elif not isinstance(x_vals, (tuple, list)):
        x_vals = (x_vals, )

    if cm_names is None:
        cm_names = ['Blues', 'Oranges', 'Greens',  'Reds', 'Purples', ]
    elif isinstance(cm_names, str):
        cm_names = [cm_names]

    cm = [getattr(plt.cm, cm_name) for cm_name in cm_names]

    if axes is None:
        fig, ax = plt.subplots(**fig_kwargs)
    else:
        ax = axes

    if not isinstance(ax, np.ndarray):
        ax = [ax]
    else:
        ax = ax.flatten()
    # The x label goes on the axes' own figure, created here or supplied.
    figure = ax[0].figure

    states_and_fstates = _state_metadata(
        getattr(uo, 'states_di', None), getattr(uo, 'fstates_di', None))

    if times is not None:
        if len(times) == 1:
            colors = [[None]] * len(cm)
        else:
            ls = np.linspace(0.2, 0.8, len(times))
            colors = [cmap(ls) for cmap in cm]

        y = get_state_distrib(uo.result, *state_names, time=times,
                              x_name=x_name)

        names = list(y.keys())
        x_vals = getattr(uo.result, x_name)

        for t, time in enumerate(times):
            for ind, name in enumerate(names):
                axis = ax[ind]
                y_plot = y[name]

                if isinstance(y_plot, list):
                    for st, ar in enumerate(y_plot):
                        ind_cm = st % len(cm)
                        axis.plot(x_vals, ar[t], color=colors[ind_cm][t])
                else:
                    axis.plot(x_vals, y_plot[t], color=colors[0][t])

        name_yaxes(ax, states_and_fstates, names, ylabels, legend)
        set_legend(ax, states_and_fstates, names, state_names, legend)

        for axis in ax:
            if len(axis.lines) == 0:
                axis.remove()

        figure.text(0.5, 0, x_name)

        if len(ax) == 1:
            ax = ax[0]

    else:
        if len(x_vals) == 1:
            colors = [[None]] * len(cm)
        else:
            ls = np.linspace(0.2, 0.8, len(x_vals))
            colors = [cmap(ls) for cmap in cm]

        y_di = get_state_distrib(uo.result, *state_names, x=x_vals,
                                 x_name=x_name)

        names = list(y_di.keys())

        time = uo.result.time
        for ct, x in enumerate(x_vals):
            for ind, (state, y) in enumerate(y_di.items()):
                if isinstance(y, list):
                    for st, ar in enumerate(y):
                        ind_cm = st % len(cm)
                        ax[ind].plot(time, ar[:, ct], color=colors[ind_cm][ct])
                else:
                    ax[ind].plot(time, y[:, ct], color=colors[0][ct])

        name_yaxes(ax, states_and_fstates, names, ylabels, legend)
        set_legend(ax, states_and_fstates, names, state_names, legend)

    if 'fig' in locals():
        return fig, ax
    else:
        return axes  # the caller's own axes object, as in plot_function
