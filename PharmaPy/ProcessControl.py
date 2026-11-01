# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:06:16 2021

@author: dcasasor
"""

from typing import Optional

# Entries of a normalized control record: the callable, its positional
# arguments after time, and its keyword arguments (see analyze_controls).
CONTROL_RECORD_FIELDS = ('fun', 'args', 'kwargs')


def analyze_controls(di: dict) -> dict:
    """Normalize named control callables and records.

    Parameters
    ----------
    di : dict
        Control names mapped to callables or records containing ``fun``.
        Callables accept time [s] and return values in the controlled units.

    Returns
    -------
    dict
        Records containing ``fun``, ``args`` and ``kwargs`` for each control.

    Raises
    ------
    KeyError
        If a control record lacks ``fun``.
    TypeError
        If the supplied function is not callable.

    Notes
    -----
    Returns new records: supplied records are copied before missing ``args``
    and ``kwargs`` default to an empty tuple and dictionary, so neither
    ``di`` nor its records are modified. The ``args`` and ``kwargs`` values
    themselves are not copied.
    """

    controls = {}

    for key, val in di.items():
        if isinstance(val, dict) and key != 'kwargs':
            if 'fun' not in val:
                raise KeyError("'%s' dictionary must have a 'fun' field" % key)
            elif not callable(val['fun']):
                raise TypeError(
                    "Object passed to the 'fun' field must be a "
                    "callable with signature fun(time, *args, **kwargs)")

            out = dict(val)
        else:
            if not callable(val):
                raise TypeError(
                    "Object passed to the 'fun' field must be a "
                    "callable with signature fun(time, *args, **kwargs)")

            out = {'fun': val}

        if 'args' not in out:
            out['args'] = ()

        if 'kwargs' not in out:
            out['kwargs'] = {}

        controls[key] = out

    return controls


def normalize_controls(controls: Optional[dict]) -> dict:
    """Normalize and validate unit control records.

    Parameters
    ----------
    controls : dict or None
        Controlled state names mapped to callables ``f(time, *args,
        **kwargs)`` or to records with ``fun`` and optional ``args`` and
        ``kwargs``. Time is [s]; the returned value is in the controlled
        state's units, such as temperature [K] or volume [m**3]. None means
        no controls.

    Returns
    -------
    dict
        New dictionary of new records with exactly the ``fun``, ``args``
        and ``kwargs`` entries; missing ``args`` and ``kwargs`` default to
        an empty tuple and dictionary. Each ``kwargs`` is a new dictionary,
        so no dictionary of the caller is shared or modified; the ``args``
        sequence and the argument values themselves are not copied.

    Raises
    ------
    ValueError
        If a record has entries other than ``fun``, ``args`` and ``kwargs``.
    KeyError
        If a record has no ``fun`` entry.
    TypeError
        If ``fun`` is not callable, ``args`` is not a tuple or list, or
        ``kwargs`` is not a dictionary.
    """
    normalized = analyze_controls(controls or {})
    for name, control in normalized.items():
        if set(control) != set(CONTROL_RECORD_FIELDS):
            raise ValueError(
                f"Control {name!r} must be callable or have exactly "
                "fun, args, kwargs keys")
        if not callable(control['fun']):
            raise TypeError(f"Control {name!r} 'fun' must be callable")
        if (not isinstance(control['args'], (tuple, list))
                or not isinstance(control['kwargs'], dict)):
            raise TypeError(
                f"Control {name!r} requires sequence args and dictionary kwargs")
        control['kwargs'] = dict(control['kwargs'])
    return normalized


def apply_control_modifier(controls: dict, modify_controls: Optional[dict],
                           owner: str) -> dict:
    """Return configured controls updated by one experiment's modifier.

    Parameters
    ----------
    controls : dict
        Configured control records of a unit, as returned by
        ``normalize_controls``.
    modify_controls : dict or None
        Configured control names mapped to partial control records.
        ``fun`` and ``args`` replace the configured entries; ``kwargs`` is
        merged key by key into the configured keyword arguments, so
        keywords it omits keep their configured values. Values are in the
        units of the control function's arguments; the control returns the
        controlled state's units (such as temperature [K]) from time [s].
        None or an empty dictionary means the configured controls.
    owner : str
        Noun naming the unit in error messages, such as ``'reactor'``.

    Returns
    -------
    dict
        New normalized control records (see ``normalize_controls``) with
        fresh ``kwargs`` dictionaries, one per configured control.
        ``controls`` and ``modify_controls`` are not modified.

    Raises
    ------
    TypeError
        If ``modify_controls``, one of its entries, or an entry's
        ``kwargs`` is not a dictionary, or the merged record has a
        non-callable ``fun`` or non-sequence ``args``.
    ValueError
        If ``modify_controls`` names a control that is not configured (the
        message lists the configured names), or an entry has keys other
        than ``fun``, ``args`` and ``kwargs``.
    """
    if modify_controls is None:
        modify_controls = {}
    if not isinstance(modify_controls, dict):
        raise TypeError(
            "modify_controls must be a dictionary mapping configured "
            "control names to partial control records, or None; got "
            f"{type(modify_controls).__name__}")
    unknown = [name for name in modify_controls if name not in controls]
    if unknown:
        raise ValueError(
            f"modify_controls names unknown controls {unknown!r}; the "
            f"{owner}'s configured controls are {list(controls)!r}")

    merged = {}
    for name, record in controls.items():
        update = modify_controls.get(name, {})
        if not isinstance(update, dict):
            raise TypeError(
                f"modify_controls[{name!r}] must be a dictionary with "
                f"entries among {list(CONTROL_RECORD_FIELDS)!r}; got "
                f"{type(update).__name__}")
        unknown_entries = [entry for entry in update
                           if entry not in CONTROL_RECORD_FIELDS]
        if unknown_entries:
            raise ValueError(
                f"modify_controls[{name!r}] has unknown entries "
                f"{unknown_entries!r}; allowed entries are "
                f"{list(CONTROL_RECORD_FIELDS)!r}")
        kwargs_update = update.get('kwargs', {})
        if not isinstance(kwargs_update, dict):
            raise TypeError(
                f"modify_controls[{name!r}]['kwargs'] must be a "
                f"dictionary; got {type(kwargs_update).__name__}")
        merged[name] = {**record, **update,
                        'kwargs': {**record['kwargs'], **kwargs_update}}

    return normalize_controls(merged)


class DynamicInput:
    def __init__(self):
        self.controls = {}
        self.args_control = {}

        # Attributes assigned from UO instance
        self.parent_instance = None

    def add_variable(self, variable_name, function, args_control=None):
        self.controls[variable_name] = function
        if args_control is None:
            args_control = ()

        self.args_control[variable_name] = args_control

    def evaluate_inputs(self, time):
        controls_out = {}

        for key, fun in self.controls.items():
            args = self.args_control[key]
            controls_out[key] = fun(time, *args)

        return controls_out
