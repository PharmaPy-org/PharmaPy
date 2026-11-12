"""Reduce the raw caffeine runs to the tables in ``reduced/``.

Usage, from the repository root::

    python -m examples.pat.caffeine.reduce_runs "<folder that holds the four run folders>"

Each run gives three tables on the clock of its temperature program: thermostat
temperatures, FBRM band counts with the square-weighted mean chord, and the height
of the caffeine IR band. ``runs.json`` holds every setting used here.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

from .. import fbrm_statistics, ic_fbrm, ic_spectra, table_layout
from ..band_height import band_height
from ..run_table import block_mean

CASE = Path(__file__).parent
REDUCED = CASE / "reduced"
SECONDS_PER_HOUR = 3600.0  # [s/h]
MEAN_CHORD = "mean_chord_square_weighted"
# Decimals written per quantity: the thermostat log resolves 0.01 K, counts are
# fractional after rebinning, and the IR band scatters by about 2e-4.
TIME_DIGITS, TEMPERATURE_DIGITS, COUNT_DIGITS, CHORD_DIGITS, HEIGHT_DIGITS = 1, 3, 1, 2, 5


def load_description() -> dict:
    """Return the dataset description of the caffeine runs.

    Returns
    -------
    dict
        Contents of ``runs.json``.
    """
    return json.loads((CASE / "runs.json").read_text(encoding="utf-8"))


def run_clock_origin(first_fbrm_sample: datetime.datetime, run: dict, description: dict) -> datetime.datetime:
    """Return the local time at which the temperature program of a run started.

    Parameters
    ----------
    first_fbrm_sample : datetime.datetime
        Time stamp of the first FBRM sample (local time, naive).
    run : dict
        Entry of the run in ``runs.json``.
    description : dict
        Dataset description; gives the FBRM sample interval [s].

    Returns
    -------
    datetime.datetime
        Zero of the run clock in local time: the start of the first FBRM sample
        interval plus the run's ``temperature_log_start_after_fbrm_start_s``.
    """
    interval = description["fbrm"]["sample_interval_s"]  # [s]
    offset = run["temperature_log_start_after_fbrm_start_s"]  # [s]
    return first_fbrm_sample + datetime.timedelta(seconds=offset - interval)


def temperature_table(log_path: Path, description: dict) -> tuple[dict, dict]:
    """Reduce a thermostat log to block means.

    Parameters
    ----------
    log_path : pathlib.Path
        Thermostat log; its time zero is the zero of the run clock.
    description : dict
        Dataset description; gives the layout file and the block size.

    Returns
    -------
    tuple of (dict, dict)
        ``{column: values}`` with ``time`` [s] and the jacket, external and
        setpoint temperatures [K], each the mean of one block, and ``{column: unit}``.
    """
    settings = description["temperature_log"]
    values, units = table_layout.read_table(log_path, table_layout.load_layout(CASE / settings["layout"]))
    columns = {name: block_mean(series, settings["block_samples"]) for name, series in values.items()}
    return columns, units


def fbrm_table(times: list[datetime.datetime], raw: np.ndarray, origin: datetime.datetime, description: dict) -> tuple[dict, dict]:
    """Reduce raw chord-length samples to band counts and a mean chord.

    Parameters
    ----------
    times : list of datetime.datetime
        Sample time stamps (local time, naive).
    raw : numpy.ndarray
        Counts on the raw grid [counts], shape ``(n_samples, 1550)``.
    origin : datetime.datetime
        Zero of the run clock in local time.
    description : dict
        Dataset description; gives the band edges [um].

    Returns
    -------
    tuple of (dict, dict)
        ``{column: values}`` with ``time`` [s], one ``counts_<lower>_to_<upper>um``
        column per band [counts], and the square-weighted mean chord [um] on the
        100-channel logarithmic grid (NaN for a sample without counts), and
        ``{column: unit}``.
    """
    raw_edges = ic_fbrm.raw_channel_edges()  # [um]
    band_edges = np.array(description["fbrm"]["band_edges_um"])  # [um]
    log_edges = fbrm_statistics.log_channel_edges(band_edges[0], band_edges[-1])  # [um]
    bands = fbrm_statistics.band_counts(raw, raw_edges, band_edges)
    log_counts = fbrm_statistics.band_counts(raw, raw_edges, log_edges)
    columns = {"time": np.array([(time - origin).total_seconds() for time in times])}
    units = {"time": "s"}
    for index in range(band_edges.size - 1):
        name = "counts_%g_to_%gum" % (band_edges[index], band_edges[index + 1])
        columns[name], units[name] = bands[:, index], "counts"
    columns[MEAN_CHORD] = fbrm_statistics.weighted_mean_chord(log_counts, fbrm_statistics.geometric_midpoints(log_edges))
    units[MEAN_CHORD] = "um"
    return columns, units


def _parse_utc(stamp: str) -> datetime.datetime:
    """Parse an iC ``SampleTime`` such as ``2026-10-24T04:29:10.2574757Z``.

    Parameters
    ----------
    stamp : str
        ISO 8601 UTC time with up to seven fractional digits.

    Returns
    -------
    datetime.datetime
        Naive UTC time, truncated to microseconds.
    """
    head, _, fraction = stamp.rstrip("Z").partition(".")
    return datetime.datetime.strptime(head + "." + (fraction[:6] or "0"), "%Y-%m-%dT%H:%M:%S.%f")


def ir_table(infos: dict, records: dict, origin: datetime.datetime, utc_offset_h: float, description: dict) -> tuple[dict, dict]:
    """Reduce IR spectra to the height of the band named in the description.

    Parameters
    ----------
    infos : dict
        Spectrum metadata from `ic_spectra.spectrum_infos`.
    records : dict
        ``{spectrum identifier: absorbance}``, each of shape ``(n_points,)``.
    origin : datetime.datetime
        Zero of the run clock in local time.
    utc_offset_h : float
        Local time minus UTC on the day of the run [h].
    description : dict
        Dataset description; gives the band and its baseline anchors [cm-1].

    Returns
    -------
    tuple of (dict, dict)
        ``{column: values}`` with ``time`` [s], ``band_height`` [-] (absorbance)
        and ``solvent_absorbance`` [-], the absorbance at the solvent band, which
        drops when the probe loses contact with the liquid; sorted by time. Then
        ``{column: unit}``.

    Raises
    ------
    ValueError
        If a spectrum has no metadata or the spectra do not share one axis.
    """
    band = description["ir_band"]
    rows, axes = [], set()
    for identifier, absorbance in records.items():
        if identifier not in infos:
            raise ValueError("Spectrum %s has no metadata entry." % identifier)
        info = infos[identifier]
        axes.add((float(info["StartWn"]), float(info["EndWn"]), int(info["NumberOfPoints"])))
        local = _parse_utc(info["SampleTime"]) + datetime.timedelta(hours=utc_offset_h)
        rows.append(((local - origin).total_seconds(), absorbance))
    if len(axes) != 1:
        raise ValueError("The spectra do not share one spectral axis: %s." % sorted(axes))
    start, end, points = axes.pop()
    axis = np.linspace(start, end, points)  # [cm-1]
    rows.sort(key=lambda row: row[0])
    spectra = np.vstack([row[1] for row in rows])
    columns = {
        "time": np.array([row[0] for row in rows]),
        "band_height": band_height(axis, spectra, band["peak_cm-1"], tuple(band["baseline_cm-1"])),
        "solvent_absorbance": spectra[:, int(np.argmin(np.abs(axis - band["solvent_position_cm-1"])))],
    }
    return columns, {"time": "s", "band_height": "-", "solvent_absorbance": "-"}


def _single(folder: Path, pattern: str) -> Path:
    """Return the only file in ``folder`` that matches ``pattern``.

    Parameters
    ----------
    folder : pathlib.Path
        Run folder.
    pattern : str
        Glob pattern.

    Returns
    -------
    pathlib.Path
        The matching file.

    Raises
    ------
    ValueError
        If no file or more than one file matches.
    """
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise ValueError("Expected one %s file in %s, found %d." % (pattern, folder, len(matches)))
    return matches[0]


def reduce_run(raw_root: Path, run: dict, description: dict, out_dir: Path) -> dict[str, int]:
    """Write the three reduced tables of one run.

    Parameters
    ----------
    raw_root : pathlib.Path
        Folder that holds the run folders.
    run : dict
        Entry of the run in ``runs.json``.
    description : dict
        Dataset description.
    out_dir : pathlib.Path
        Folder for ``<run id>_temperature.csv``, ``<run id>_fbrm.csv`` and
        ``<run id>_ir.csv``.

    Returns
    -------
    dict
        Number of rows written per table [-].
    """
    folder = raw_root / run["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    times, raw = ic_fbrm.read_samples(_single(folder, "*.icFbrm"))
    origin = run_clock_origin(times[0], run, description)
    tables = {
        "temperature": temperature_table(folder / description["temperature_log"]["file"], description),
        "fbrm": fbrm_table(times, raw, origin, description),
        "ir": ir_table(*ic_spectra.load_spectra(_single(folder, "*.iCIR")), origin, run["fbrm_local_time_utc_offset_h"], description),
    }
    by_unit = {"s": TIME_DIGITS, "K": TEMPERATURE_DIGITS, "counts": COUNT_DIGITS, "um": CHORD_DIGITS, "-": HEIGHT_DIGITS}
    rows = {}
    for kind, (columns, units) in tables.items():
        digits = {name: by_unit[unit] for name, unit in units.items()}
        table_layout.write_table(out_dir / ("%s_%s.csv" % (run["id"], kind)), columns, units, digits, missing=(MEAN_CHORD,))
        rows[kind] = columns["time"].size
    return rows


def main(arguments: list[str]) -> None:
    """Reduce every run listed in ``runs.json``.

    Parameters
    ----------
    arguments : list of str
        The folder that holds the run folders, then optionally the output folder
        (default ``reduced/`` beside this file).
    """
    description = load_description()
    out_dir = Path(arguments[1]) if len(arguments) > 1 else REDUCED
    for run in description["runs"]:
        print(run["id"], reduce_run(Path(arguments[0]), run, description, out_dir))


if __name__ == "__main__":
    main(sys.argv[1:])
