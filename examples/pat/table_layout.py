"""Read delimited text tables through an explicit layout description.

A layout names the columns a file must have, their units and their plausible range.
The reader finds columns by header name, so column order does not matter, and it
refuses a file that does not match instead of guessing.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# File unit -> (unit of the returned values, factor, offset): value_out = factor * value + offset.
# Every entry is an exact definition (60 s per min, 3600 s per h, 0 C = 273.15 K).
UNIT_CONVERSIONS = {
    "s": ("s", 1.0, 0.0),
    "min": ("s", 60.0, 0.0),
    "h": ("s", 3600.0, 0.0),
    "K": ("K", 1.0, 0.0),
    "C": ("K", 1.0, 273.15),
    "-": ("-", 1.0, 0.0),
}
ROLES = ("time", "value")
COLUMN_KEYS = {"header", "unit", "role", "minimum", "maximum"}
LAYOUT_KEYS = {"description", "delimiter", "header_row", "units_row", "first_data_row", "columns", "range_basis"}


@dataclass(frozen=True)
class ColumnLayout:
    """One required column of a table.

    Attributes
    ----------
    name : str
        Name of the returned field.
    header : str
        Column title in the file, compared after stripping surrounding blanks.
    unit : str
        Unit of the values in the file; a key of `UNIT_CONVERSIONS`.
    role : str
        ``"time"`` for the single strictly increasing time column, else ``"value"``.
    minimum, maximum : float
        Inclusive plausible range in the file's unit; infinite when not limited.
    """

    name: str
    header: str
    unit: str
    role: str = "value"
    minimum: float = -math.inf
    maximum: float = math.inf


@dataclass(frozen=True)
class TableLayout:
    """Layout of a delimited text table.

    Attributes
    ----------
    columns : tuple of ColumnLayout
        Required columns.
    delimiter : str
        Cell separator.
    header_row : int
        Zero-based row that holds the column titles.
    units_row : int or None
        Zero-based row that holds the unit of each column, when the file has one.
    first_data_row : int
        Zero-based row of the first data record.
    description : str
        What the file is and where it comes from.
    """

    columns: tuple[ColumnLayout, ...]
    delimiter: str = ","
    header_row: int = 0
    units_row: int | None = None
    first_data_row: int = 1
    description: str = ""


def load_layout(path: str | Path) -> TableLayout:
    """Read a layout description from a JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON file with the keys of `TableLayout`; ``columns`` maps each field name
        to the keys of `ColumnLayout`. ``range_basis`` is free text that states
        why the column ranges were chosen.

    Returns
    -------
    TableLayout
        The validated layout.

    Raises
    ------
    ValueError
        If a key, unit or role is unknown, or the layout does not have exactly one
        time column.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    unknown = set(raw) - LAYOUT_KEYS
    if unknown or "columns" not in raw:
        raise ValueError("Layout %s: unknown keys %s or missing 'columns'." % (path, sorted(unknown)))
    columns = []
    for name, entry in raw["columns"].items():
        extra = set(entry) - COLUMN_KEYS
        if extra or "header" not in entry or "unit" not in entry:
            raise ValueError("Layout %s, column %r: needs 'header' and 'unit'; unknown keys %s." % (path, name, sorted(extra)))
        column = ColumnLayout(name=name, **entry)
        if column.unit not in UNIT_CONVERSIONS:
            raise ValueError("Layout %s, column %r: unit %r is not one of %s." % (path, name, column.unit, sorted(UNIT_CONVERSIONS)))
        if column.role not in ROLES:
            raise ValueError("Layout %s, column %r: role %r is not one of %s." % (path, name, column.role, ROLES))
        columns.append(column)
    if sum(column.role == "time" for column in columns) != 1:
        raise ValueError("Layout %s must declare exactly one column with role 'time'." % path)
    options = {key: raw[key] for key in ("delimiter", "header_row", "units_row", "first_data_row", "description") if key in raw}
    return TableLayout(columns=tuple(columns), **options)


def write_table(path: str | Path, columns: dict[str, np.ndarray], units: dict[str, str], digits: dict[str, int]) -> None:
    """Write columns as a comma-separated table with a title row and a unit row.

    Parameters
    ----------
    path : str or pathlib.Path
        File to write.
    columns : dict
        ``{column title: values}`` with each array of shape ``(n_rows,)``, written
        in this order.
    units : dict
        ``{column title: unit}``; written as the second row.
    digits : dict
        ``{column title: number of decimals}`` [-] for each column.

    Raises
    ------
    ValueError
        If the columns differ in length, hold a non-finite value, or lack a unit
        or a number of decimals.
    """
    names = list(columns)
    arrays = [np.asarray(columns[name], dtype=float) for name in names]
    if any(name not in units or name not in digits for name in names):
        raise ValueError("Every column needs a unit and a number of decimals.")
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 1:
        raise ValueError("Columns must be one-dimensional and of equal length; got %s." % [array.shape for array in arrays])
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("A column holds a non-finite value; a reduced table must be complete.")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        writer.writerow([units[name] for name in names])
        for row in zip(*arrays):
            writer.writerow(["%.*f" % (digits[name], value) for name, value in zip(names, row)])


def read_table(path: str | Path, layout: TableLayout) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read the columns a layout declares.

    Parameters
    ----------
    path : str or pathlib.Path
        Delimited text file.
    layout : TableLayout
        Layout the file must match.

    Returns
    -------
    tuple of (dict, dict)
        ``{field name: values}`` with each array of shape ``(n_rows,)``, converted
        with `UNIT_CONVERSIONS` (times in [s], temperatures in [K]), and
        ``{field name: unit of the returned values}``.

    Raises
    ------
    ValueError
        If a declared column is missing or duplicated, a unit in the file differs
        from the layout, a cell is not a finite number, a value is outside its
        declared range, the time column does not increase strictly, or the file
        has no data rows.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle, delimiter=layout.delimiter) if any(cell.strip() for cell in row)]
    if len(rows) <= layout.first_data_row:
        raise ValueError("%s has no data rows at or after row %d." % (path, layout.first_data_row))
    header = [cell.strip() for cell in rows[layout.header_row]]
    positions = {}
    for column in layout.columns:
        if header.count(column.header) != 1:
            raise ValueError("%s: expected exactly one column titled %r, found %d. Titles present: %s."
                             % (path, column.header, header.count(column.header), header))
        positions[column.name] = header.index(column.header)
        if layout.units_row is not None:
            file_unit = rows[layout.units_row][positions[column.name]].strip()
            if file_unit != column.unit:
                raise ValueError("%s: column %r is in %r but the layout declares %r." % (path, column.header, file_unit, column.unit))
    values, units = {}, {}
    for column in layout.columns:
        position = positions[column.name]
        try:
            raw = np.array([float(row[position]) for row in rows[layout.first_data_row:]])
        except (ValueError, IndexError) as error:
            raise ValueError("%s: column %r holds a missing or non-numeric cell (%s)." % (path, column.header, error)) from error
        if not np.all(np.isfinite(raw)):
            raise ValueError("%s: column %r holds a non-finite value." % (path, column.header))
        if raw.min() < column.minimum or raw.max() > column.maximum:
            raise ValueError("%s: column %r spans %g to %g %s, outside the declared range %g to %g."
                             % (path, column.header, raw.min(), raw.max(), column.unit, column.minimum, column.maximum))
        if column.role == "time" and not np.all(np.diff(raw) > 0):
            raise ValueError("%s: time column %r does not increase strictly." % (path, column.header))
        units[column.name], factor, offset = UNIT_CONVERSIONS[column.unit]
        values[column.name] = factor * raw + offset
    return values, units
