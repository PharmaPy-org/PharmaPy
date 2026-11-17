"""Access to a described dataset: its description and its reduced tables.

A dataset is a folder with a description ``runs.json``, one layout
``reduced_<kind>.layout.json`` per kind of reduced table, and the tables
``reduced/<run>_<kind>.csv``. Solute and solvent are named in the description.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import table_layout

SIGNIFICANT_DIGITS = 6  # of stored results; a design choice, finer than the scatter of the signals


def load_description(case: Path) -> dict:
    """Return the description of a dataset.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.

    Returns
    -------
    dict
        Contents of its ``runs.json``.
    """
    return json.loads((Path(case) / "runs.json").read_text(encoding="utf-8"))


def read_reduced(case: Path, run_id: str, kind: str) -> dict:
    """Read one reduced table of a run through its layout.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    run_id : str
        Run identifier from the description.
    kind : str
        Kind of table, for example ``"temperature"`` or ``"ir"``.

    Returns
    -------
    dict
        ``{column: values}`` in the units the layout declares after conversion:
        times in [s], temperatures in [K].
    """
    case = Path(case)
    layout = table_layout.load_layout(case / ("reduced_%s.layout.json" % kind))
    return table_layout.read_table(case / "reduced" / ("%s_%s.csv" % (run_id, kind)), layout)[0]


def charge_ratio(description: dict) -> float:
    """Return the solute charged per solvent.

    Parameters
    ----------
    description : dict
        Dataset description with ``charge.solute_mass_g`` and
        ``charge.solvent_mass_g`` [g].

    Returns
    -------
    float
        Solute per solvent [kg/kg].

    Raises
    ------
    ValueError
        If either mass is not positive.
    """
    solute = description["charge"]["solute_mass_g"]  # [g]
    solvent = description["charge"]["solvent_mass_g"]  # [g]
    if solute <= 0.0 or solvent <= 0.0:
        raise ValueError("Solute (%g g) and solvent (%g g) masses of the charge must be positive." % (solute, solvent))
    return solute / solvent


def rounded(value):
    """Round every float of a nested result to `SIGNIFICANT_DIGITS` significant digits.

    Parameters
    ----------
    value : dict, float, int or str
        A result or one of its parts.

    Returns
    -------
    dict, float, int or str
        The same structure with rounded floats.
    """
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, float):
        return float("%.*g" % (SIGNIFICANT_DIGITS, value))
    return value


def write_result(case: Path, name: str, result: dict) -> Path:
    """Write a result into a dataset folder as JSON, with rounded floats.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    name : str
        File name of the result.
    result : dict
        Nested result of strings and numbers.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    path = Path(case) / name
    path.write_text(json.dumps(rounded(result), indent=2) + "\n", encoding="utf-8", newline="\n")
    return path
