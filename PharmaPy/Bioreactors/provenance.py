"""Portable forward-case records aligned with the requirements of WP1 #351.

These versioned JSON dictionaries adopt the existing observation adapter. They
are not the pending Experiment/Measurement object API (#346). Missing samples
remain omitted on each field's own grid; no masks, weights or fitted values are
invented. Explicit orders survive JSON tools that reorder object keys.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from PharmaPy.DataClasses import StateCollection
from .observations import read_observations


SCHEMA = "pharmapy-bioreactor-forward-record/1"


# Record integrity helpers -----------------------------------------------------
def _digest(value: Mapping) -> str:
    """Hash JSON input content independently of mapping insertion order.

    Parameters
    ----------
    value : Mapping
        JSON-compatible input; scientific values retain their declared units.

    Returns
    -------
    str
        SHA256 of canonical JSON, excluding no fields.

    Raises
    ------
    ValueError
        A numeric value is nonfinite.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _ordered_keys(values: Mapping, order: list, label: str) -> None:
    """Require an explicit, unique order covering a mapping exactly.

    Parameters
    ----------
    values : Mapping
        Named records.
    order : list of str
        Declared order.
    label : str
        Name used in validation errors.

    Raises
    ------
    ValueError
        Names are empty, duplicated, missing or unexpected.
    """
    if (not isinstance(values, Mapping) or not isinstance(order, list)
            or any(not isinstance(name, str) or not name for name in order)
            or len(set(order)) != len(order) or set(order) != set(values)):
        raise ValueError(f"{label} order must cover its named entries exactly")


def _parameter_values(parameters: Mapping, prefix: str = "") -> dict:
    """Flatten input parameter groups while preserving their declared order.

    Parameters
    ----------
    parameters : Mapping
        Named scalar model parameters, in input-declared units.
    prefix : str, optional
        Dotted parent path, used during recursive group traversal.

    Returns
    -------
    dict
        Dotted parameter paths to finite scalar values in their original bases.

    Raises
    ------
    ValueError
        A parameter is not a finite scalar; this forward record version does
        not infer units or shapes for vector-valued parameter groups.
    """
    values = {}
    for name, value in parameters.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            values.update(_parameter_values(value, path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            values[path] = value  # [input-declared parameter unit]
        else:
            raise ValueError(f"parameter {path!r} must be a finite scalar")
    return values


# Validation keeps identities and physical bases explicit. ---------------------
def validate_record(record: Mapping) -> None:
    """Validate the forward record without changing order, values or weights.

    Parameters
    ----------
    record : Mapping
        Record from ``make_record`` or decoded JSON. Samples have shape (n,)
        per field, independent times [s], and units/bases in their declaration.

    Raises
    ------
    ValueError
        Identity, input hashes, field alignment, parameter metadata or optional
        predictions are inconsistent. Unknown schemas are rejected.

    Notes
    -----
    This adapter accepts finite observations on possibly unequal grids. Missing
    samples are omitted, never imputed. Duplicate and nonmonotone sample order is
    preserved. Uncertainty metadata is stored, never converted into fit weights.
    """
    if not isinstance(record, Mapping) or record.get("schema") != SCHEMA:
        raise ValueError("unsupported forward provenance record schema")
    if not isinstance(record.get("experiment_id"), str) or not record["experiment_id"]:
        raise ValueError("experiment_id must be a nonempty string")
    model = record.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("model metadata must be a mapping")
    if not isinstance(model.get("revision"), str) or not model["revision"]:
        raise ValueError("model revision must be declared explicitly")
    inputs = record.get("inputs", {})
    if not isinstance(inputs, Mapping) or set(inputs) != {"case", "mechanism", "thermo", "observations"}:
        raise ValueError("record requires case, mechanism, thermo and observations inputs")
    if model.get("input_sha256") != {key: _digest(value) for key, value in inputs.items()}:
        raise ValueError("record input hashes do not match its input snapshots")
    if model.get("solver_settings") != inputs["case"]["numerics"]:
        raise ValueError("solver settings must match the recorded case")
    if record.get("source_observations_sha256") != _digest(record["source_observations"]):
        raise ValueError("source observation hash does not match the stored observations")
    metadata = record.get("provenance", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("provenance metadata must be a mapping")
    for key in ("sources", "sharing_conditions", "transformations", "uncertainty", "calibration"):
        if key not in metadata:
            raise ValueError(f"provenance requires {key}")
    if not metadata["sources"] or not metadata["sharing_conditions"]:
        raise ValueError("data sources and sharing conditions must be explicit")
    if metadata["calibration"] != {"status": "not_performed", "fitted_values": None}:
        raise ValueError("forward records cannot claim a local calibration")
    parameters = record.get("parameters", {})
    _ordered_keys(parameters, record.get("parameter_order"), "parameter")
    coefficients = inputs["mechanism"].get("parameters", {})
    if set(parameters) != set(_parameter_values(coefficients)):
        raise ValueError("parameter metadata must cover every input parameter exactly")
    for name, entry in parameters.items():
        value = coefficients
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(f"parameter {name!r} is absent from the mechanism")
            value = value[part]
        if not entry.get("unit") or not entry.get("basis") or "bounds" not in entry:
            raise ValueError(f"parameter {name!r} needs unit, basis and bounds metadata")
        if entry.get("value") != value:
            raise ValueError(f"parameter {name!r} value disagrees with the mechanism")
        bounds = entry["bounds"]
        if bounds is None:
            if not entry.get("bounds_status"):
                raise ValueError(f"parameter {name!r} needs a reason for unspecified bounds")
        elif (len(bounds) != 2 or not np.isfinite(bounds).all()
              or not bounds[0] <= value <= bounds[1]):
            raise ValueError(f"parameter {name!r} has inconsistent bounds")
    fields = inputs["observations"]["fields"]
    order = record.get("field_order")
    _ordered_keys(fields, order, "measurement")
    declaration = dict(inputs["observations"], fields={name: fields[name] for name in order})
    observed = read_observations(declaration, record["source_observations"])
    for name, field in fields.items():
        for key in (field["time_key"], field["value_key"]):
            if np.asarray(record["source_observations"][name][key]).ndim != 1:
                raise ValueError(f"measurement {name!r} requires one-dimensional sample vectors")
    if record.get("missing_data") != "omitted_samples":
        raise ValueError("forward records require the omitted_samples missing-data policy")
    predictions = record.get("predictions")
    if predictions is not None:
        for key in ("time_s", "values", "units"):
            _ordered_keys(predictions.get(key, {}), order, f"prediction {key}")
        for name in order:
            values = np.asarray(predictions["values"][name], dtype=float)  # [field unit]
            if (values.shape != observed["values"][name].shape or not np.isfinite(values).all()
                    or predictions["units"][name] != observed["units"][name]
                    or not np.array_equal(predictions["time_s"][name], observed["time_s"][name])):
                raise ValueError(f"prediction {name!r} must match observation grid, unit and shape")
    for segment in record.get("native_outputs", []):
        times = np.asarray(segment["time_s"], dtype=float)  # [s]
        _ordered_keys(segment["fields"], segment["field_order"], "native output")
        if times.ndim != 1 or not times.size or not np.isfinite(times).all():
            raise ValueError("native output times must be a finite vector in seconds")
        for name, field in segment["fields"].items():
            values = np.asarray(field["values"], dtype=float)  # [field unit]
            if (field["kind"] not in {"state", "algebraic", "quality_attribute"}
                    or not field["unit"] or not field["basis"]
                    or list(values.shape) != field["shape"]
                    or values.ndim < 1 or values.shape[0] != len(times)
                    or not np.isfinite(values).all()):
                raise ValueError(f"native output {name!r} has inconsistent identity, units or shape")
    # Reject NaN/Infinity anywhere, including metadata, before writing portable JSON.
    _digest(record)


# Public snapshots and adapters preserve the existing native inputs. -----------
def make_record(experiment_id: str, case: Mapping, mechanism: Mapping,
                thermo: Mapping, declaration: Mapping, data: Mapping, *,
                provenance: Mapping, parameter_metadata: Mapping,
                model_revision: str, predictions: Mapping = None, native_outputs=None) -> dict:
    """Snapshot one forward case and its observation provenance for replay.

    Parameters
    ----------
    experiment_id : str
        Stable experiment identity, distinct from the model revision.
    case, mechanism, thermo : Mapping
        Existing builder inputs, with their original units and physical bases.
    declaration : Mapping
        Existing observation declaration, in authoritative field order.
    data : Mapping
        Source observations consumed by ``read_observations``; values are not
        transformed again. Each field has its own sample vector.
    provenance : Mapping
        Sources/revisions, sharing conditions, transformations already applied,
        uncertainty meaning, and calibration status. This forward adapter requires
        ``calibration={'status': 'not_performed', 'fitted_values': None}``.
    parameter_metadata : Mapping
        Dotted paths below mechanism ``parameters``, in declared parameter order;
        each entry gives unit, basis and bounds (or null plus bounds_status).
        Values are read from the actual mechanism, not copied from metadata.
    model_revision : str
        Explicit code revision/identifier; input hashes identify the exact inputs.
        A dirty revision must be identified as such by the caller.
    predictions : Mapping, optional
        Output of ``predict_measurements``. Stored as lists without changing
        time [s], field order, shape or observation units.

    native_outputs : list of Mapping, optional
        Segmented native fields from ``native_output_fields``; time [s], units,
        component identity, field kind and array shapes remain explicit.

    Returns
    -------
    dict
        Independent JSON-compatible record. It does not run a model, fit
        parameters, infer measurement uncertainty or write any files.

    Raises
    ------
    ValueError
        Metadata, parameter paths or predictions are inconsistent.
    """
    parameters = deepcopy(dict(parameter_metadata))
    for name, entry in parameters.items():
        value = mechanism.get("parameters", {})
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(f"parameter {name!r} is absent from the mechanism")
            value = value[part]
        entry["value"] = value  # [entry unit], same physical basis as input
    inputs = {"case": case, "mechanism": mechanism, "thermo": thermo,
              "observations": declaration}
    record = {"schema": SCHEMA, "experiment_id": experiment_id,
              "field_order": list(declaration["fields"]),
              "parameter_order": list(parameters), "parameters": parameters,
              "inputs": deepcopy(inputs), "source_observations": deepcopy(data),
              "source_observations_sha256": _digest(data),
              "native_outputs": deepcopy(native_outputs) if native_outputs is not None else [],
              "missing_data": "omitted_samples", "provenance": deepcopy(provenance),
              "model": {"revision": model_revision,
                        "input_sha256": {key: _digest(value) for key, value in inputs.items()},
                        "solver_settings": deepcopy(case["numerics"])},
              "predictions": None}
    if predictions is not None:
        record["predictions"] = {
            "time_s": {name: np.asarray(value).tolist() for name, value in predictions["time_s"].items()},
            "values": {name: np.asarray(value).tolist() for name, value in predictions["values"].items()},
            "units": dict(predictions["units"])}
    validate_record(record)
    return record


def native_output_fields(unit, histories) -> list:
    """Copy native results with field identity, component order, units and shape.

    Parameters
    ----------
    unit : MultiPhaseVessel
        Solved native unit supplying state metadata.
    histories : Sequence[DynamicResult]
        Accepted native result segments, including both sides of recipe events.

    Returns
    -------
    list of dict
        One record per segment with time_s [s] and named fields. Differential
        fields are states; algebraic constraints and postprocessed expressions
        are algebraic fields. No quality attribute is invented when none is
        declared by the model. The basis identifies native state units and phase;
        species/component labels retain metadata order.

    Raises
    ------
    AttributeError
        A declared native field is missing from a supplied history.
    """
    metadata = {**unit.solver_state_collection.states, **unit.output_state_collection.states}
    segments = []
    for history in histories:
        fields = {}
        for key, state in metadata.items():
            name = StateCollection.format_key(key)
            values = np.asarray(getattr(history, name), dtype=float)  # [state.units]
            fields[name] = {"kind": "state" if state.state_type == "diff" else "algebraic",
                            "unit": state.units, "basis": "native " + state.name,
                            "phase": ({"type": key.phaseref.phase_type, "index": key.phaseref.index}
                                      if key.phaseref is not None else None),
                            "components": np.asarray(state.index).tolist() if state.index is not None else None,
                            "shape": list(values.shape), "values": values.tolist()}
        segments.append({"time_s": np.asarray(history.time).tolist(),
                         "field_order": list(fields), "fields": fields})
    return segments


def record_observations(record: Mapping) -> dict:
    """Adapt a record back to the existing keyed observation arrays.

    Parameters
    ----------
    record : Mapping
        Valid forward provenance record; stored field_order is authoritative.

    Returns
    -------
    dict
        Existing ``time_s``, ``values`` and ``units`` mappings, copied in declared
        field order. Arrays have shape (n_field_samples,), with time in [s].

    Raises
    ------
    ValueError
        The record is inconsistent.
    """
    validate_record(record)
    declaration = deepcopy(record["inputs"]["observations"])
    declaration["fields"] = {name: declaration["fields"][name] for name in record["field_order"]}
    return read_observations(declaration, record["source_observations"])


def write_record(record: Mapping, path) -> None:
    """Write a validated record as portable JSON to an explicit destination.

    Parameters
    ----------
    record : Mapping
        Forward record; scientific values retain their declared units.
    path : path-like
        Caller-selected output file. Generated records belong outside source inputs.

    Raises
    ------
    ValueError
        Record validation fails.
    OSError
        The destination cannot be written.
    """
    validate_record(record)
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_record(path) -> dict:
    """Read and validate a JSON record without executing its content.

    Parameters
    ----------
    path : path-like
        Previously written forward record.

    Returns
    -------
    dict
        Validated record; explicit orders are retained independently of JSON keys.

    Raises
    ------
    ValueError
        JSON or record validation fails.
    OSError
        The source cannot be read.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_record(record)
    return record
