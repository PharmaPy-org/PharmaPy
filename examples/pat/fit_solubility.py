"""Solubility curve of a described dataset, from spectra taken while solid is present.

Usage, from the repository root::

    python -m examples.pat.fit_solubility "<dataset folder>"

Reads the ``solubility`` block of the dataset description, the band calibration
(``band_calibration.json``) and the reduced tables, and writes ``solubility.json``
into the dataset folder. The block gives a time window in which the solid is
dissolving on a heating ramp, and a solid signal with a threshold that tells
whether solid is still present. Each spectrum in the window with solid present
gives one point: the dissolved solute per solvent at its temperature.

Dissolution lags the heating, so these points are a lower bound of the
solubility. The curve is fitted per run and over all runs together, under each
treatment of the band calibration. The solubility is one physical curve, so the
scatter of the pooled fit shows how well the runs agree under a treatment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import calibrate_band, dataset, solubility

RESULT_NAME = "solubility.json"


def saturated_points(case: Path, run_id: str, treatment: str, calibration: dict, description: dict
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return the points of a run taken while solid is present.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    run_id : str
        A run with a band calibration.
    treatment : str
        Treatment of the band calibration.
    calibration : dict
        Contents of ``band_calibration.json``.
    description : dict
        Dataset description, with its ``solubility`` and ``band_calibration``
        blocks.

    Returns
    -------
    temperature : numpy.ndarray
        Temperatures of the points [K], shape ``(n_points,)``.
    ratio : numpy.ndarray
        Dissolved solute per solvent [kg/kg], shape ``(n_points,)``.

    Notes
    -----
    The solid signal is the sum of the listed columns, interpolated at the
    spectrum times. A point is kept when that sum is at least
    ``minimum_solid_signal``.
    """
    settings = description["solubility"]
    time, ratio = calibrate_band.solute_ratio(case, run_id, treatment, calibration)
    temperature = calibrate_band.spectra_with_temperature(case, run_id, description["band_calibration"])[2]  # [K]
    solid = dataset.read_reduced(case, run_id, settings["solid_table"])
    signal = np.interp(time, solid["time"], sum(solid[column] for column in settings["solid_columns"]))
    start, stop = settings["window_s"]  # [s]
    kept = (time >= start) & (time <= stop) & (signal >= settings["minimum_solid_signal"])
    return temperature[kept], ratio[kept]


def _curve(temperature: np.ndarray, ratio: np.ndarray, charge: float) -> dict:
    """Fit one curve and describe it.

    Parameters
    ----------
    temperature : numpy.ndarray
        Temperatures of the points [K], shape ``(n_points,)``.
    ratio : numpy.ndarray
        Dissolved solute per solvent [kg/kg], shape ``(n_points,)``.
    charge : float
        Solute charged per solvent [kg/kg].

    Returns
    -------
    dict
        ``intercept`` [-], ``slope_K`` [K], ``scatter`` [-] (see
        `solubility.fit_vant_hoff`), ``count``, the fitted range
        ``minimum_temperature_K`` and ``maximum_temperature_K`` [K], and
        ``charge_saturation_temperature_K``, the temperature at which the curve
        equals the charge [K].
    """
    intercept, slope, scatter = solubility.fit_vant_hoff(temperature, ratio)
    return {"intercept": intercept, "slope_K": slope, "scatter": scatter, "count": int(temperature.size),
            "minimum_temperature_K": float(temperature.min()), "maximum_temperature_K": float(temperature.max()),
            "charge_saturation_temperature_K": float(solubility.saturation_temperature(charge, intercept, slope))}


def fit(case: Path) -> dict:
    """Fit the solubility curve of a dataset under every treatment of its band calibration.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder, holding ``band_calibration.json``.

    Returns
    -------
    dict
        ``solute`` and ``solvent`` (names), ``charge_ratio`` [kg/kg] and
        ``treatments``: per treatment a ``pooled`` curve over all calibrated runs
        and one curve per run under ``runs`` (see `_curve` for a curve's entries).
    """
    case = Path(case)
    description = dataset.load_description(case)
    calibration = json.loads((case / calibrate_band.RESULT_NAME).read_text(encoding="utf-8"))
    charge = calibration["charge_ratio"]  # [kg/kg]
    treatments = {}
    for treatment in calibrate_band.TREATMENTS:
        points = {run_id: saturated_points(case, run_id, treatment, calibration, description)
                  for run_id in calibration["runs"]}
        treatments[treatment] = {
            "pooled": _curve(np.concatenate([temperature for temperature, _ in points.values()]),
                             np.concatenate([ratio for _, ratio in points.values()]), charge),
            "runs": {run_id: _curve(temperature, ratio, charge) for run_id, (temperature, ratio) in points.items()},
        }
    return {"solute": calibration["solute"], "solvent": calibration["solvent"], "charge_ratio": charge,
            "treatments": treatments}


def main(arguments: list[str]) -> None:
    """Write ``solubility.json`` into a dataset folder.

    Parameters
    ----------
    arguments : list of str
        The dataset folder.
    """
    case = Path(arguments[0])
    print("wrote", dataset.write_result(case, RESULT_NAME, fit(case)))


if __name__ == "__main__":
    main(sys.argv[1:])
