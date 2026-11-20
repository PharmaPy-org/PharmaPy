"""Reduce the raw runs of a described dataset to the tables in its ``reduced/`` folder.

Usage, from the repository root::

    python -m examples.pat.reduce_runs "<dataset folder>" "<folder that holds the run folders>"

Each run gives tables on the clock of its temperature program: thermostat
temperatures, FBRM band counts with the square-weighted mean chord, and an IR
signal of the dissolved solute. The description names that signal in one of two
blocks: ``ir_band``, the height of one band, or ``ir_score``, the whole-spectrum
score of `spectral_score`, which needs the spectra of all runs together.
``runs.json`` of the dataset holds every setting used here.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np

from . import dataset, fbrm_statistics, ic_fbrm, ic_spectra, spectral_score, table_layout
from .band_height import band_height
from .run_table import block_mean

SECONDS_PER_HOUR = 3600.0  # [s/h]
MEAN_CHORD = "mean_chord_square_weighted"
# Decimals written per quantity: the thermostat log resolves 0.01 K, counts are
# fractional after rebinning, and the IR band scatters by about 2e-4.
TIME_DIGITS, TEMPERATURE_DIGITS, COUNT_DIGITS, CHORD_DIGITS, HEIGHT_DIGITS = 1, 3, 1, 2, 5
DIGITS_BY_UNIT = {"s": TIME_DIGITS, "K": TEMPERATURE_DIGITS, "counts": COUNT_DIGITS, "um": CHORD_DIGITS, "-": HEIGHT_DIGITS}
IR_BLOCKS = ("ir_band", "ir_score")
CLOCK_KEYS = ("temperature_log_start_local", "temperature_log_start_after_fbrm_start_s")
SCORE_SUMMARY = "ir_score.json"


def run_clock_origin(first_fbrm_sample: datetime.datetime | None, run: dict, description: dict) -> datetime.datetime:
    """Return the local time at which the temperature program of a run started.

    Parameters
    ----------
    first_fbrm_sample : datetime.datetime or None
        Time stamp of the first FBRM sample (local time, naive); None for a run
        without FBRM samples.
    run : dict
        Entry of the run in ``runs.json``. It places the clock in one of two ways:
        ``temperature_log_start_local``, the local time at which the thermostat log
        started (ISO 8601), or ``temperature_log_start_after_fbrm_start_s``, its
        start after the start of the first FBRM sample interval [s].
    description : dict
        Dataset description; gives the FBRM sample interval [s].

    Returns
    -------
    datetime.datetime
        Zero of the run clock in local time.

    Raises
    ------
    ValueError
        If the run gives both ways or neither, or refers to the first FBRM sample
        without having one.
    """
    absolute, relative = CLOCK_KEYS
    if (absolute in run) == (relative in run):
        raise ValueError("Run %s must give exactly one of %s and %s." % (run["id"], absolute, relative))
    if absolute in run:
        return datetime.datetime.fromisoformat(run[absolute])
    if first_fbrm_sample is None:
        raise ValueError("Run %s places its clock by the first FBRM sample but has no FBRM samples." % run["id"])
    interval = description["fbrm"]["sample_interval_s"]  # [s]
    offset = run["temperature_log_start_after_fbrm_start_s"]  # [s]
    return first_fbrm_sample + datetime.timedelta(seconds=offset - interval)


def temperature_table(log_path: Path, description: dict, case: Path) -> tuple[dict, dict]:
    """Reduce a thermostat log to block means.

    Parameters
    ----------
    log_path : pathlib.Path
        Thermostat log; its time zero is the zero of the run clock.
    description : dict
        Dataset description; gives the layout file and the block size.
    case : pathlib.Path
        Dataset folder, which holds the layout file.

    Returns
    -------
    tuple of (dict, dict)
        ``{column: values}`` with ``time`` [s] and the jacket, external and
        setpoint temperatures [K], each the mean of one block, and ``{column: unit}``.
    """
    settings = description["temperature_log"]
    values, units = table_layout.read_table(log_path, table_layout.load_layout(Path(case) / settings["layout"]))
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


def timed_spectra(infos: dict, records: dict, origin: datetime.datetime, utc_offset_h: float,
                  without_metadata: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put the spectra of one file on the run clock.

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
    without_metadata : int, optional
        Number of spectra the file is known to hold without a metadata entry [-].
        They have no sample time and are left out. Default 0.

    Returns
    -------
    time : numpy.ndarray
        Sample times on the run clock [s], increasing, shape ``(n_spectra,)``.
    axis : numpy.ndarray
        Spectral axis [cm-1], shape ``(n_points,)``.
    spectra : numpy.ndarray
        Absorbance [-], shape ``(n_spectra, n_points)``, in the order of ``time``.

    Raises
    ------
    ValueError
        If the number of spectra without metadata differs from the declared one,
        or the spectra do not share one axis.
    """
    orphans = [identifier for identifier in records if identifier not in infos]
    if len(orphans) != without_metadata:
        raise ValueError("%d spectra without metadata were declared, but %d were found%s."
                         % (without_metadata, len(orphans), "; %s has no metadata entry" % orphans[0] if orphans else ""))
    rows, axes = [], set()
    for identifier, absorbance in records.items():
        if identifier not in infos:
            continue
        info = infos[identifier]
        axes.add((float(info["StartWn"]), float(info["EndWn"]), int(info["NumberOfPoints"])))
        local = _parse_utc(info["SampleTime"]) + datetime.timedelta(hours=utc_offset_h)
        rows.append(((local - origin).total_seconds(), absorbance))
    if len(axes) != 1:
        raise ValueError("The spectra do not share one spectral axis: %s." % sorted(axes))
    start, end, points = axes.pop()
    axis = np.linspace(start, end, points)  # [cm-1]
    rows.sort(key=lambda row: row[0])
    return np.array([row[0] for row in rows]), axis, np.vstack([row[1] for row in rows])


def ir_table(infos: dict, records: dict, origin: datetime.datetime, utc_offset_h: float, description: dict,
             without_metadata: int = 0) -> tuple[dict, dict]:
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
    without_metadata : int, optional
        Number of spectra known to lack a metadata entry [-]. Default 0.

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
        If the spectra without metadata differ from the declared number, or the
        spectra do not share one axis.
    """
    band = description["ir_band"]
    time, axis, spectra = timed_spectra(infos, records, origin, utc_offset_h, without_metadata)
    columns = {
        "time": time,
        "band_height": band_height(axis, spectra, band["peak_cm-1"], tuple(band["baseline_cm-1"])),
        "solvent_absorbance": spectra[:, int(np.argmin(np.abs(axis - band["solvent_position_cm-1"])))],
    }
    return columns, {"time": "s", "band_height": "-", "solvent_absorbance": "-"}


def _in_window(time: np.ndarray, window: list[float], what: str, run_id: str) -> np.ndarray:
    """Return the samples of a run inside a time window.

    Parameters
    ----------
    time : numpy.ndarray
        Sample times on the run clock [s], shape ``(n_samples,)``.
    window : list of float
        First and last time [s], both included.
    what : str
        Name of the window, for the error message.
    run_id : str
        Run identifier, for the error message.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If no sample lies inside the window.
    """
    inside = (time >= window[0]) & (time <= window[1])
    if not inside.any():
        raise ValueError("Run %s has no spectrum in the %s window, %g to %g s." % (run_id, what, window[0], window[1]))
    return inside


def score_tables(spectra_by_run: dict, description: dict) -> tuple[dict, dict]:
    """Reduce the IR spectra of all runs to the whole-spectrum score.

    Parameters
    ----------
    spectra_by_run : dict
        ``{run id: (time, temperature, axis, spectra)}``: sample times on the run
        clock [s] and liquid temperatures [K], both of shape ``(n_spectra,)``, the
        spectral axis [cm-1] of shape ``(n_points,)``, and absorbance [-] of shape
        ``(n_spectra, n_points)``.
    description : dict
        Dataset description. Its ``ir_score`` block names the reference run and
        its two windows on the run clock, ``clear_solution_window_s`` (one
        composition at several temperatures) and ``slurry_window_s``; the window
        ``dissolved_window_s`` in which every run has its whole charge dissolved;
        the reference temperature [K]; the spectral window [cm-1] with the largest
        absorbance kept [-]; and the comparison run.

    Returns
    -------
    tables : dict
        ``{run id: ({column: values}, {column: unit})}`` with ``time`` [s] and,
        all [-], ``score``, ``scale``, ``offset``, ``tilt`` and ``residual`` of
        `spectral_score.score`. The score is zero for the run's own dissolved
        charge and minus one for a drop of dissolved solute equal to the one
        between the dissolved charge and the slurry of the reference run.
    summary : dict
        Sizes of the fit, and the terms of every run's dissolved spectrum against
        the comparison run's: the difference of their charges in score units.

    Raises
    ------
    ValueError
        If the runs do not share one spectral axis or a window holds no spectrum.
    """
    settings = description["ir_score"]
    reference_run, comparison_run = settings["reference_run"], settings["comparison_run"]
    reference_temperature = settings["reference_temperature_K"]  # [K]
    time, temperature, axis, spectra = spectra_by_run[reference_run]
    for run_id, item in spectra_by_run.items():
        if item[2].shape != axis.shape or not np.array_equal(item[2], axis):
            raise ValueError("Run %s does not share the spectral axis of run %s." % (run_id, reference_run))
    clear = _in_window(time, settings["clear_solution_window_s"], "clear-solution", reference_run)
    level, per_kelvin, effect_residual = spectral_score.temperature_effect(spectra[clear], temperature[clear], reference_temperature)
    lower, upper = settings["wavenumber_window_cm-1"]
    keep = spectral_score.usable_points(axis, level, lower, upper, settings["maximum_absorbance"])
    moved, dissolved, counts = {}, {}, {}
    for run_id, (run_time, run_temperature, _, run_spectra) in spectra_by_run.items():
        moved[run_id] = spectral_score.at_reference_temperature(run_spectra, run_temperature, per_kelvin, reference_temperature)[:, keep]
        inside = _in_window(run_time, settings["dissolved_window_s"], "dissolved", run_id)
        dissolved[run_id], counts[run_id] = moved[run_id][inside].mean(axis=0), int(inside.sum())
    slurry = _in_window(time, settings["slurry_window_s"], "slurry", reference_run)
    direction = dissolved[reference_run] - moved[reference_run][slurry].mean(axis=0)  # [-]
    units = {name: "-" for name in ("score", "scale", "offset", "tilt", "residual")}
    units = {"time": "s", **units}
    tables, against, condition = {}, {}, None
    for run_id, item in spectra_by_run.items():
        terms = spectral_score.score(axis[keep], moved[run_id], dissolved[run_id], direction)
        tables[run_id] = ({"time": item[0], "score": terms.score, "scale": terms.scale, "offset": terms.offset,
                           "tilt": terms.tilt, "residual": terms.residual}, dict(units))
        compared = spectral_score.score(axis[keep], dissolved[run_id][None, :], dissolved[comparison_run], direction)
        against[run_id] = {"score": float(compared.score[0]), "scale": float(compared.scale[0]), "offset": float(compared.offset[0]),
                           "tilt": float(compared.tilt[0]), "residual": float(compared.residual[0]), "dissolved_spectra": counts[run_id]}
        condition = terms.condition if run_id == reference_run else condition
    summary = {
        "reference_run": reference_run, "comparison_run": comparison_run,
        "spectral_points": int(keep.sum()), "clear_solution_spectra": int(clear.sum()), "slurry_spectra": int(slurry.sum()),
        "clear_solution_temperature_min_K": float(temperature[clear].min()), "clear_solution_temperature_max_K": float(temperature[clear].max()),
        "temperature_effect_residual": effect_residual, "condition": condition,
        "dissolved_spectrum_against_comparison_run": against,
    }
    return tables, summary


def spectra_with_temperature(time: np.ndarray, axis: np.ndarray, spectra: np.ndarray, temperature_time: np.ndarray,
                             temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Give every spectrum the liquid temperature at its sample time.

    Parameters
    ----------
    time : numpy.ndarray
        Sample times of the spectra on the run clock [s], shape ``(n_spectra,)``.
    axis : numpy.ndarray
        Spectral axis [cm-1], shape ``(n_points,)``.
    spectra : numpy.ndarray
        Absorbance [-], shape ``(n_spectra, n_points)``.
    temperature_time : numpy.ndarray
        Times of the temperature table [s], increasing, shape ``(n_rows,)``.
    temperature : numpy.ndarray
        Liquid temperature at those times [K], shape ``(n_rows,)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(time, temperature, axis, spectra)`` of the spectra taken while the
        temperature was logged, the temperature interpolated linearly. Spectra
        before the first or after the last row of the table are left out.
    """
    logged = (time >= temperature_time[0]) & (time <= temperature_time[-1])
    return time[logged], np.interp(time[logged], temperature_time, temperature), axis, spectra[logged]


def _write_tables(out_dir: Path, run_id: str, tables: dict) -> dict[str, int]:
    """Write reduced tables of one run and return their row counts.

    Parameters
    ----------
    out_dir : pathlib.Path
        Folder for ``<run id>_<kind>.csv``.
    run_id : str
        Run identifier.
    tables : dict
        ``{kind: ({column: values}, {column: unit})}``.

    Returns
    -------
    dict
        Number of rows written per kind [-].
    """
    rows = {}
    for kind, (columns, units) in tables.items():
        digits = {name: DIGITS_BY_UNIT[unit] for name, unit in units.items()}
        table_layout.write_table(out_dir / ("%s_%s.csv" % (run_id, kind)), columns, units, digits, missing=(MEAN_CHORD,))
        rows[kind] = columns["time"].size
    return rows


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


def reduce_run(case: Path, raw_root: Path, run: dict, description: dict, out_dir: Path) -> tuple[dict[str, int], tuple | None]:
    """Write the reduced tables of one run that need no other run.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    raw_root : pathlib.Path
        Folder that holds the run folders.
    run : dict
        Entry of the run in ``runs.json``.
    description : dict
        Dataset description.
    out_dir : pathlib.Path
        Folder for ``<run id>_temperature.csv``, ``<run id>_fbrm.csv`` and, for a
        dataset with an ``ir_band`` block, ``<run id>_ir.csv``.

    Returns
    -------
    rows : dict
        Number of rows written per table [-].
    spectra : tuple or None
        For a dataset with an ``ir_score`` block, the input of `score_tables` for
        this run: ``(time, temperature, axis, spectra)``. Otherwise None.

    Raises
    ------
    ValueError
        If the description does not hold exactly one of the two IR blocks, or
        ``has_fbrm_samples`` of the run is not a boolean.

    Notes
    -----
    A run with ``"has_fbrm_samples": false`` gets no FBRM table and its FBRM file
    is not opened. ``ir_spectra_without_metadata`` of a run declares spectra that
    the file holds without a sample time.
    """
    if sum(block in description for block in IR_BLOCKS) != 1:
        raise ValueError("The description must hold exactly one of the blocks %s." % (IR_BLOCKS,))
    has_fbrm = run.get("has_fbrm_samples", True)
    if not isinstance(has_fbrm, bool):
        raise ValueError("has_fbrm_samples of run %s must be true or false, not %r." % (run["id"], has_fbrm))
    folder = raw_root / run["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    times, raw = ic_fbrm.read_samples(_single(folder, "*.icFbrm")) if has_fbrm else (None, None)
    origin = run_clock_origin(times[0] if has_fbrm else None, run, description)
    tables = {"temperature": temperature_table(folder / description["temperature_log"]["file"], description, case)}
    if has_fbrm:
        tables["fbrm"] = fbrm_table(times, raw, origin, description)
    infos, records = ic_spectra.load_spectra(_single(folder, "*.iCIR"))
    utc_offset_h, orphans = run["fbrm_local_time_utc_offset_h"], run.get("ir_spectra_without_metadata", 0)
    spectra = None
    if "ir_band" in description:
        tables["ir"] = ir_table(infos, records, origin, utc_offset_h, description, orphans)
    else:
        settings = description["ir_score"]
        log = tables["temperature"][0]
        spectra = spectra_with_temperature(*timed_spectra(infos, records, origin, utc_offset_h, orphans),
                                           log["time"], log[settings["temperature_column"]])
    return _write_tables(out_dir, run["id"], tables), spectra


def main(arguments: list[str]) -> None:
    """Reduce every run listed in the description of a dataset.

    Parameters
    ----------
    arguments : list of str
        The dataset folder, the folder that holds the run folders, then optionally
        the output folder (default ``reduced/`` in the dataset folder).

    Notes
    -----
    For a dataset with an ``ir_score`` block the IR tables are written after all
    runs are read, together with ``ir_score.json`` in the output folder.
    """
    case = Path(arguments[0])
    description = dataset.load_description(case)
    out_dir = Path(arguments[2]) if len(arguments) > 2 else case / "reduced"
    rows, spectra_by_run = {}, {}
    for run in description["runs"]:
        rows[run["id"]], spectra = reduce_run(case, Path(arguments[1]), run, description, out_dir)
        if spectra is not None:
            spectra_by_run[run["id"]] = spectra
    if spectra_by_run:
        tables, summary = score_tables(spectra_by_run, description)
        for run_id, table in tables.items():
            rows[run_id].update(_write_tables(out_dir, run_id, {"ir": table}))
        dataset.write_result(out_dir, SCORE_SUMMARY, summary)
    for run_id, counts in rows.items():
        print(run_id, counts)


if __name__ == "__main__":
    main(sys.argv[1:])
