"""Band calibration of a described dataset: band height to dissolved solute per solvent.

Usage, from the repository root::

    python -m examples.pat.calibrate_band "<dataset folder>"

Reads the ``band_calibration`` block of the dataset description and the reduced
tables, and writes ``band_calibration.json`` into the dataset folder. The block
names the table and columns to use, the runs, the reference run (the one with
solvent-only spectra) and the time windows of known composition.

When the dissolved plateaus of the runs differ at the same nominal charge, the
data cannot tell a shifted baseline from a different slope, so both treatments
are computed:

- ``common_baseline``: the solvent baseline of the reference run for every run,
  and a slope per run from its own dissolved plateaus.
- ``common_slope``: the slope of the reference run for every run, and a baseline
  per run that reproduces its own dissolved plateaus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from PharmaPy import PATCalibration

from . import dataset

TREATMENTS = ("common_baseline", "common_slope")
RESULT_NAME = "band_calibration.json"
DISSOLVED = ("dissolved_first", "dissolved_second")


def spectra_with_temperature(case: Path, run_id: str, settings: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the band heights of a run that have a logged temperature.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    run_id : str
        Run identifier from the description.
    settings : dict
        The ``band_calibration`` block of the description.

    Returns
    -------
    time : numpy.ndarray
        Spectrum times on the run clock [s], shape ``(n_spectra,)``.
    band_height : numpy.ndarray
        Band heights [-], shape ``(n_spectra,)``.
    temperature : numpy.ndarray
        Temperature interpolated at the spectrum times [K], shape ``(n_spectra,)``.

    Notes
    -----
    Spectra taken before the temperature log starts or after it ends are left out,
    so no temperature is extrapolated.
    """
    spectra = dataset.read_reduced(case, run_id, settings["table"])
    log = dataset.read_reduced(case, run_id, settings["temperature_table"])
    logged = (spectra["time"] >= log["time"][0]) & (spectra["time"] <= log["time"][-1])
    time = spectra["time"][logged]  # [s]
    return time, spectra[settings["height_column"]][logged], np.interp(time, log["time"], log[settings["temperature_column"]])


def solvent_baseline(case: Path, settings: dict) -> dict:
    """Return the band height of the solvent alone, from the reference run.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    settings : dict
        The ``band_calibration`` block of the description.

    Returns
    -------
    dict
        ``band_height`` (median), ``minimum`` and ``maximum`` [-] and ``count`` of
        the spectra in the solvent window.

    Notes
    -----
    No temperature is attached to these spectra: the baseline is taken as
    independent of temperature. The minimum and maximum bound it.
    """
    spectra = dataset.read_reduced(case, settings["reference_run"], settings["table"])
    height = spectra[settings["height_column"]]
    start, stop = settings["solvent_window_s"]  # [s]
    median, _, count = PATCalibration.window_statistics(spectra["time"], height, start, stop)
    inside = height[(spectra["time"] >= start) & (spectra["time"] <= stop)]
    return {"band_height": median, "minimum": float(inside.min()), "maximum": float(inside.max()), "count": count}


def plateaus(case: Path, run_id: str, settings: dict) -> dict:
    """Return the band height and temperature of the calibration windows of a run.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    run_id : str
        Run identifier from the description.
    settings : dict
        The ``band_calibration`` block of the description.

    Returns
    -------
    dict
        ``{window: {"band_height", "scatter", "count", "temperature_K"}}`` for
        ``dissolved_first`` and ``dissolved_second`` (the two windows of
        ``dissolved_windows_s``, in that order), ``slurry_start`` and
        ``slurry_end``. Band height is the median [-], scatter its standard
        deviation [-], temperature the mean over the spectra [K].
    """
    time, height, temperature = spectra_with_temperature(case, run_id, settings)
    windows = dict(zip(DISSOLVED, settings["dissolved_windows_s"]))
    windows["slurry_start"] = settings["slurry_start_window_s"]
    windows["slurry_end"] = [time[-1] - settings["slurry_end_duration_s"], time[-1]]
    result = {}
    for name, (start, stop) in windows.items():
        median, scatter, count = PATCalibration.window_statistics(time, height, start, stop)
        inside = (time >= start) & (time <= stop)
        result[name] = {"band_height": median, "scatter": scatter, "count": count,
                        "temperature_K": float(temperature[inside].mean())}
    return result


def calibrate(case: Path) -> dict:
    """Compute the calibration constants of every listed run under both treatments.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.

    Returns
    -------
    dict
        ``solute`` and ``solvent`` (names), ``charge_ratio`` [kg/kg],
        ``reference_temperature_K``, ``solvent_baseline`` (see `solvent_baseline`)
        and ``runs``: per run its ``plateaus`` and, per treatment, ``baseline``
        [-], ``slope`` [-] per [kg/kg], ``slope_per_kelvin`` [1/K] per [kg/kg],
        and the dissolved solute per solvent of the two slurry windows,
        ``slurry_start_ratio`` and ``slurry_end_ratio`` [kg/kg].

    Notes
    -----
    The dissolved windows are taken to hold the whole charge in solution.
    """
    description = dataset.load_description(case)
    settings = description["band_calibration"]
    reference_temperature = settings["reference_temperature_K"]  # [K]
    ratio = dataset.charge_ratio(description)  # [kg/kg]
    solvent = solvent_baseline(case, settings)
    windows = {run_id: plateaus(case, run_id, settings) for run_id in settings["runs"]}

    def dissolved(run_id):
        """Return the heights [-] and temperatures [K] of the two dissolved plateaus of a run."""
        pair = [windows[run_id][name] for name in DISSOLVED]
        return [entry["band_height"] for entry in pair], [entry["temperature_K"] for entry in pair]

    reference_slope = PATCalibration.fit_band_slope(*dissolved(settings["reference_run"]), ratio, solvent["band_height"],
                                                 reference_temperature)
    runs = {}
    for run_id in settings["runs"]:
        heights, temperatures = dissolved(run_id)
        constants = {
            "common_baseline": (solvent["band_height"],) + PATCalibration.fit_band_slope(
                heights, temperatures, ratio, solvent["band_height"], reference_temperature),
            "common_slope": (PATCalibration.fit_band_baseline(heights, temperatures, ratio, *reference_slope,
                                                           reference_temperature),) + reference_slope,
        }
        runs[run_id] = {"plateaus": windows[run_id]}
        for treatment, (baseline, slope, slope_per_kelvin) in constants.items():
            entry = {"baseline": baseline, "slope": slope, "slope_per_kelvin": slope_per_kelvin}
            for name in ("slurry_start", "slurry_end"):
                window = windows[run_id][name]
                entry[name + "_ratio"] = float(PATCalibration.ratio_from_band_height(
                    window["band_height"], window["temperature_K"], baseline, slope, slope_per_kelvin,
                    reference_temperature))
            runs[run_id][treatment] = entry
    return {"solute": description["charge"]["solute"], "solvent": description["charge"]["solvent"],
            "charge_ratio": ratio, "reference_temperature_K": reference_temperature, "solvent_baseline": solvent,
            "runs": runs}


def solute_ratio(case: Path, run_id: str, treatment: str, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the dissolved solute per solvent of a run over time.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    run_id : str
        A run listed under ``band_calibration`` in the description.
    treatment : str
        ``"common_baseline"`` or ``"common_slope"``.
    calibration : dict
        Result of `calibrate`, or the contents of ``band_calibration.json``.

    Returns
    -------
    time : numpy.ndarray
        Spectrum times on the run clock [s], shape ``(n_spectra,)``.
    ratio : numpy.ndarray
        Dissolved solute per solvent [kg/kg], shape ``(n_spectra,)``.

    Raises
    ------
    ValueError
        If the run has no calibration or the treatment is unknown.

    Notes
    -----
    The band measures whatever absorbs at the probe. While solid sits on the probe
    window, or another solid form of the solute is still dissolving, it is not a
    measure of the dissolved solute alone.
    """
    if run_id not in calibration["runs"]:
        raise ValueError("Run %r has no band calibration; calibrated runs are %s." % (run_id, sorted(calibration["runs"])))
    if treatment not in TREATMENTS:
        raise ValueError("Unknown treatment %r; choose one of %s." % (treatment, list(TREATMENTS)))
    constants = calibration["runs"][run_id][treatment]
    settings = dataset.load_description(case)["band_calibration"]
    time, height, temperature = spectra_with_temperature(case, run_id, settings)
    return time, PATCalibration.ratio_from_band_height(
        height, temperature, constants["baseline"], constants["slope"], constants["slope_per_kelvin"],
        calibration["reference_temperature_K"])


def main(arguments: list[str]) -> None:
    """Write ``band_calibration.json`` into a dataset folder.

    Parameters
    ----------
    arguments : list of str
        The dataset folder.
    """
    case = Path(arguments[0])
    print("wrote", dataset.write_result(case, RESULT_NAME, calibrate(case)))


if __name__ == "__main__":
    main(sys.argv[1:])
