"""Map native bioreactor states to declared measured fields.

A measurement declaration names each observed field, the phase species it
measures, the reported unit and, at a scheduled event time, whether the sample
represents the state before or after that event. Observed values stay in their
source files. Predictions are read from accepted solver states at the exact
sampling times (see ``NativeBioreactorAssembly.solve(report_times=...)``); no
interpolation between reported grid points is performed.

Output mappings are keyed by field in declaration order, with times in [s], so
one experiment's ``time_s`` and ``values`` can be passed as the per-field
``x_data``/``y_data`` entries of the keyed ``ParameterEstimation`` interface.
This module defines no experiment or measurement object hierarchy; that belongs
to the shared estimation work (#346).
"""

from collections.abc import Mapping, Sequence
from typing import Optional, TYPE_CHECKING

import numpy as np

from .BioreactorUnitConverter import BioreactorUnitConverter

if TYPE_CHECKING:
    from .construction import NativeBioreactorAssembly

# Declaration layout ----------------------------------------------------------
_DECLARATION_FIELDS = {"schema_version", "time_unit", "fields"}
_FIELD_KEYS = {"species", "quantity", "unit", "event_side", "time_key", "value_key"}
_QUANTITIES = {"concentration", "temperature"}
_EVENT_SIDES = {"pre", "post"}


# Validation helpers ------------------------------------------------------------
def _validate_declaration(declaration: Mapping,
                          species_names: Optional[Sequence[str]] = None) -> dict:
    """Validate a measurement declaration and return its fields in order.

    Parameters
    ----------
    declaration : Mapping
        ``schema_version`` (1), ``time_unit`` (a ``TIME`` unit of the source
        sampling times) and ``fields``: an ordered mapping from field name to
        ``species``, ``quantity`` (``'concentration'`` or ``'temperature'``), ``unit`` (an
        amount-per-volume unit such as ``'g/L'`` or ``'mmol/L'``),
        ``event_side`` (``'pre'`` or ``'post'``), and the source-file column
        names ``time_key`` and ``value_key``. Temperature uses ``species: null``
        and unit ``K`` or ``degC``. A ``$comment`` entry is ignored.
    species_names : Sequence[str], optional
        Phase species; when given, every concentration field must measure one
        of them. Temperature fields describe the liquid as a whole.

    Returns
    -------
    dict
        Field declarations keyed by field name, in declaration order.

    Raises
    ------
    ValueError
        Keys are missing or unknown, the schema version, time unit, quantity,
        unit or event side is unsupported, or a species is absent from the
        phase.
    """
    if not isinstance(declaration, Mapping):
        raise ValueError("measurement declaration must be a mapping")
    keys = set(declaration) - {"$comment"}
    if keys != _DECLARATION_FIELDS:
        raise ValueError(f"measurement declaration requires exactly {sorted(_DECLARATION_FIELDS)}")
    if declaration["schema_version"] != 1:
        raise ValueError("unsupported measurement schema_version")
    if declaration["time_unit"] not in BioreactorUnitConverter.TIME:
        raise ValueError("measurement time_unit must be s, min, h or day")
    fields = declaration["fields"]
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("measurement fields must be a nonempty mapping")
    result = {}
    for name, field in fields.items():
        if not isinstance(name, str) or not name:
            raise ValueError("measurement field names must be nonempty strings")
        if not isinstance(field, Mapping) or set(field) != _FIELD_KEYS:
            raise ValueError(f"measurement field {name!r} requires exactly {sorted(_FIELD_KEYS)}")
        if field["quantity"] not in _QUANTITIES:
            raise ValueError(f"measurement field {name!r} quantity must be concentration or temperature")
        if field["event_side"] not in _EVENT_SIDES:
            raise ValueError(f"measurement field {name!r} event_side must be pre or post")
        if field["quantity"] == "temperature":
            if field["species"] is not None or field["unit"] not in {"K", "degC"}:
                raise ValueError("temperature fields require species null and unit K or degC")
        else:
            if species_names is not None and field["species"] not in species_names:
                raise ValueError(f"measurement field {name!r} measures unknown species {field['species']!r}")
            # Probe the unit once; the molecular weight only enables amount bases.
            BioreactorUnitConverter.convert(1., "kg/m3", field["unit"], molecular_weight_g_mol=1.)
        result[name] = dict(field)
    return result


# Public measurement path -------------------------------------------------------
def read_observations(declaration, data):
    """Read declared measured fields from a source mapping without reordering.

    Parameters
    ----------
    declaration : Mapping
        Measurement declaration (see ``_validate_declaration``).
    data : Mapping
        Source observations keyed by field name; each entry holds the
        declared ``time_key`` (in ``time_unit``) and ``value_key`` (in the
        field's ``unit``) sequences, such as the ``measurements`` entry of a
        reference file.

    Returns
    -------
    dict
        ``time_s`` and ``values`` (field -> ndarray of sampling times [s] and
        observed values in the declared unit, in source order) and ``units``
        (field -> unit string), all keyed in declaration order.

    Raises
    ------
    ValueError
        The declaration is invalid, a declared field or column is missing,
        sequences are empty or of unequal length, or an entry is nonfinite.
        Missing observations must be omitted from the source rather than
        encoded as NaN, so every returned value is a real observation.
    """
    fields = _validate_declaration(declaration)
    time_unit = declaration["time_unit"]
    times, values, units = {}, {}, {}
    for name, field in fields.items():
        entry = data.get(name) if isinstance(data, Mapping) else None
        if not isinstance(entry, Mapping) or not {field["time_key"], field["value_key"]} <= set(entry):
            raise ValueError(f"observations lack field {name!r} or its declared columns")
        sampled = np.asarray(entry[field["time_key"]], dtype=float).reshape(-1)  # [time_unit]
        observed = np.asarray(entry[field["value_key"]], dtype=float).reshape(-1)  # [field unit]
        if not sampled.size or sampled.shape != observed.shape:
            raise ValueError(f"observations for {name!r} need equal nonempty time and value lists")
        if not np.isfinite(sampled).all() or not np.isfinite(observed).all():
            raise ValueError(f"observations for {name!r} must be finite; omit missing samples")
        times[name] = BioreactorUnitConverter.convert(sampled, time_unit, "s")  # [s]
        if np.ndim(times[name]) == 0:
            times[name] = np.array([times[name]])
        values[name] = observed
        units[name] = field["unit"]
    return {"time_s": times, "values": values, "units": units}


def sampling_times(observations):
    """Return every distinct sampling time, for ``solve(report_times=...)``.

    Parameters
    ----------
    observations : Mapping
        Result of ``read_observations``.

    Returns
    -------
    ndarray
        Sorted distinct sampling times [s].
    """
    return np.unique(np.concatenate([np.asarray(t, dtype=float) for t in
                                     observations["time_s"].values()]))


def predict_measurements(assembly: "NativeBioreactorAssembly", histories: Sequence,
                         declaration: Mapping, observations: Mapping) -> dict:
    """Predict each declared measured field at its own sampling times.

    Parameters
    ----------
    assembly : NativeBioreactorAssembly
        Solved assembly; supplies phase species and molecular weights [g/mol].
    histories : Sequence
        Native results from ``assembly.solve``, solved with at least
        ``sampling_times(observations)`` as report times.
    declaration : Mapping
        Measurement declaration.
    observations : Mapping
        Result of ``read_observations``; only ``time_s`` is used.

    Returns
    -------
    dict
        ``time_s`` (copied sampling times [s]), ``values`` (predicted values in
        each field's declared unit, aligned with ``time_s``) and ``units``, all
        keyed in declaration order.

    Raises
    ------
    ValueError
        The declaration is invalid for the assembly's phase, or the
        observations do not cover exactly the declared fields.
    RuntimeError
        A sampling time is absent from the solution, which means the assembly
        was solved without these report times.

    Notes
    -----
    A concentration is the species inventory divided by the recorded liquid
    volume, ``mass_j / vessel_vol`` [kg/m3], converted to the declared unit
    with the phase molecular weight. At an event time the solution holds two
    states; ``event_side`` selects the last state before (``'pre'``) or the
    first state after (``'post'``) the event.
    Temperature fields read native ``global_temp`` [K], with the exact
    Celsius offset 273.15 K when ``unit`` is ``degC``.
    """
    names = list(assembly.phase.name_species)
    fields = _validate_declaration(declaration, names)
    if list(observations["time_s"]) != list(fields):
        raise ValueError("observations must cover exactly the declared fields in order")
    times = np.concatenate([np.asarray(h.time, dtype=float) for h in histories])  # [s]
    masses = np.concatenate([np.asarray(h.mass_j_liquid0, dtype=float) for h in histories])  # [kg]
    volumes = np.concatenate([np.asarray(h.vessel_vol, dtype=float).reshape(-1)
                              for h in histories])  # [m3]
    predicted = {}
    for name, field in fields.items():
        rows = []
        for time in observations["time_s"][name]:
            tolerance = 8. * np.finfo(float).eps * max(1., abs(time))  # [s]; 8 ulps covers unit-conversion roundoff
            matches = np.flatnonzero(np.abs(times - time) <= tolerance)
            if not matches.size:
                raise RuntimeError(f"sampling time {time} s for {name!r} is absent from the solution; "
                                   "solve with report_times=sampling_times(observations)")
            rows.append(matches[0] if field["event_side"] == "pre" else matches[-1])
        if field["quantity"] == "temperature":
            temperatures = np.concatenate([np.asarray(h.global_temp, dtype=float).reshape(-1)
                                           for h in histories])  # [K]
            celsius_offset = 273.15 if field["unit"] == "degC" else 0.0  # [K]; exact Celsius definition
            predicted[name] = temperatures[rows] - celsius_offset  # [field unit]
        else:
            index = names.index(field["species"])
            concentration = masses[rows, index] / volumes[rows]  # [kg/m3]
            predicted[name] = BioreactorUnitConverter.convert(
                concentration, "kg/m3", field["unit"],
                molecular_weight_g_mol=assembly.phase.mw[index])  # [field unit]
        predicted[name] = np.atleast_1d(predicted[name])
    return {"time_s": {name: np.array(observations["time_s"][name], dtype=float)
                       for name in fields},
            "values": predicted,
            "units": {name: fields[name]["unit"] for name in fields}}
