"""Calibration of a process-analytical-technology signal to dissolved solute per solvent.

Two relations, each with a few constants solved from moments of known composition.
``ratio`` is the dissolved solute per solvent [kg/kg].

Band height, linear in the solute with a slope that depends on temperature::

    height = baseline + (slope + slope_per_kelvin * (T - T_ref)) * ratio

Whole-spectrum score of `PharmaPy.PATSignals.SpectralScore`, already at one
temperature and counted from the dissolved charge of its run::

    ratio = charge + sensitivity * score

Nothing here is specific to one compound.
"""
from __future__ import annotations

import numpy as np


def window_statistics(time: np.ndarray, values: np.ndarray, start: float, stop: float) -> tuple[float, float, int]:
    """Return the median and scatter of the samples inside a time window.

    Parameters
    ----------
    time : numpy.ndarray
        Sample times [s], shape ``(n_samples,)``.
    values : numpy.ndarray
        Samples, shape ``(n_samples,)``.
    start, stop : float
        Window limits [s], both included.

    Returns
    -------
    median : float
        Median of the samples, in the unit of ``values``.
    scatter : float
        Their standard deviation, in the unit of ``values``.
    count : int
        Number of samples in the window.

    Raises
    ------
    ValueError
        If the window holds no sample.
    """
    time = np.asarray(time, dtype=float)
    inside = np.asarray(values, dtype=float)[(time >= start) & (time <= stop)]
    if inside.size == 0:
        raise ValueError("No sample between %g and %g s." % (start, stop))
    return float(np.median(inside)), float(np.std(inside)), int(inside.size)


def _band_sensitivity(temperature, slope: float, slope_per_kelvin: float, reference_temperature: float):
    """Return the band height per unit solute ratio at a temperature.

    Parameters
    ----------
    temperature : float or numpy.ndarray
        Temperature [K].
    slope : float
        Band height per solute ratio at the reference temperature [-] per [kg/kg].
    slope_per_kelvin : float
        Change of that slope with temperature [1/K] per [kg/kg].
    reference_temperature : float
        Reference temperature [K].

    Returns
    -------
    float or numpy.ndarray
        Band height per solute ratio [-] per [kg/kg], shape of ``temperature``.
    """
    return slope + slope_per_kelvin * (np.asarray(temperature, dtype=float) - reference_temperature)


def band_height_from_ratio(ratio, temperature, baseline: float, slope: float, slope_per_kelvin: float,
                           reference_temperature: float):
    """Predict the band height of a solution.

    Parameters
    ----------
    ratio : float or numpy.ndarray
        Dissolved solute per solvent [kg/kg].
    temperature : float or numpy.ndarray
        Temperature of the liquid [K], broadcastable against ``ratio``.
    baseline : float
        Band height of the solvent alone [-].
    slope : float
        Band height per solute ratio at the reference temperature [-] per [kg/kg].
    slope_per_kelvin : float
        Change of that slope with temperature [1/K] per [kg/kg].
    reference_temperature : float
        Reference temperature [K].

    Returns
    -------
    numpy.ndarray
        Band height [-], broadcast shape of ``ratio`` and ``temperature``.
    """
    return baseline + _band_sensitivity(temperature, slope, slope_per_kelvin, reference_temperature) * np.asarray(ratio, dtype=float)


def ratio_from_band_height(height, temperature, baseline: float, slope: float, slope_per_kelvin: float,
                           reference_temperature: float):
    """Convert a measured band height to dissolved solute per solvent.

    Parameters
    ----------
    height : float or numpy.ndarray
        Measured band height [-].
    temperature : float or numpy.ndarray
        Temperature of the liquid [K], broadcastable against ``height``.
    baseline, slope, slope_per_kelvin, reference_temperature : float
        Calibration constants, as in `band_height_from_ratio`.

    Returns
    -------
    numpy.ndarray
        Dissolved solute per solvent [kg/kg], broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If the band height per solute ratio is not positive at some temperature,
        where the calibration cannot be inverted.
    """
    sensitivity = _band_sensitivity(temperature, slope, slope_per_kelvin, reference_temperature)
    if np.any(sensitivity <= 0.0):
        raise ValueError("The calibration slope is not positive at %g K; it cannot be inverted there."
                         % np.asarray(temperature, dtype=float)[np.asarray(sensitivity) <= 0.0].flat[0])
    return (np.asarray(height, dtype=float) - baseline) / sensitivity


def fit_band_slope(heights, temperatures, ratio: float, baseline: float, reference_temperature: float) -> tuple[float, float]:
    """Solve the slope and its temperature coefficient from two plateaus of one composition.

    Parameters
    ----------
    heights : sequence of 2 floats
        Band heights of the two plateaus [-].
    temperatures : sequence of 2 floats
        Their temperatures [K]; they must differ.
    ratio : float
        Dissolved solute per solvent on both plateaus [kg/kg]; positive.
    baseline : float
        Band height of the solvent alone [-].
    reference_temperature : float
        Reference temperature [K].

    Returns
    -------
    slope : float
        Band height per solute ratio at the reference temperature [-] per [kg/kg].
    slope_per_kelvin : float
        Change of that slope with temperature [1/K] per [kg/kg].

    Raises
    ------
    ValueError
        If there are not exactly two plateaus, their temperatures coincide, or
        ``ratio`` is not positive.
    """
    heights = np.asarray(heights, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    if heights.shape != (2,) or temperatures.shape != (2,):
        raise ValueError("Exactly two plateaus are needed; got %d heights and %d temperatures." % (heights.size, temperatures.size))
    if temperatures[0] == temperatures[1]:
        raise ValueError("The two plateaus are at the same temperature (%g K)." % temperatures[0])
    if ratio <= 0.0:
        raise ValueError("The solute ratio of the plateaus must be positive; got %g." % ratio)
    sensitivities = (heights - baseline) / ratio  # [-] per [kg/kg]
    slope_per_kelvin = (sensitivities[1] - sensitivities[0]) / (temperatures[1] - temperatures[0])
    slope = sensitivities[0] - slope_per_kelvin * (temperatures[0] - reference_temperature)
    return float(slope), float(slope_per_kelvin)


def fit_band_baseline(heights, temperatures, ratio: float, slope: float, slope_per_kelvin: float,
                 reference_temperature: float) -> float:
    """Return the baseline that makes a known slope reproduce plateaus of one composition.

    Parameters
    ----------
    heights : sequence of floats
        Band heights of the plateaus [-], shape ``(n_plateaus,)``.
    temperatures : sequence of floats
        Their temperatures [K], shape ``(n_plateaus,)``.
    ratio : float
        Dissolved solute per solvent on the plateaus [kg/kg].
    slope, slope_per_kelvin, reference_temperature : float
        Slope constants, as in `band_height_from_ratio`.

    Returns
    -------
    float
        Baseline [-]: the mean of the plateau heights minus their solute term.

    Raises
    ------
    ValueError
        If the shapes disagree or no plateau is given.
    """
    heights = np.asarray(heights, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    if heights.ndim != 1 or heights.shape != temperatures.shape or heights.size == 0:
        raise ValueError("Heights %s and temperatures %s must be equal, non-empty one-dimensional shapes."
                         % (heights.shape, temperatures.shape))
    return float(np.mean(heights - _band_sensitivity(temperatures, slope, slope_per_kelvin, reference_temperature) * ratio))


def _checked_sensitivity(sensitivity: float) -> float:
    """Return the score sensitivity after checking that it is positive.

    Parameters
    ----------
    sensitivity : float
        Dissolved solute per solvent per unit of score [kg/kg].

    Returns
    -------
    float
        The sensitivity [kg/kg].

    Raises
    ------
    ValueError
        If the sensitivity is not a positive number.
    """
    if not sensitivity > 0.0:
        raise ValueError("The score sensitivity must be positive; got %g kg/kg per score unit." % sensitivity)
    return float(sensitivity)


def fit_score_sensitivity(ratios, scores) -> float:
    """Solve the score sensitivity from two states of known composition.

    Parameters
    ----------
    ratios : sequence of 2 floats
        Dissolved solute per solvent of the two states [kg/kg].
    scores : sequence of 2 floats
        Their scores [-], counted from one common zero; they must differ. States
        of different runs are put on one zero by adding the score of each run's
        dissolved charge against that zero.

    Returns
    -------
    float
        Dissolved solute per solvent per unit of score [kg/kg].

    Raises
    ------
    ValueError
        If there are not exactly two states, their scores coincide, or the state
        with more dissolved solute does not have the higher score.

    Notes
    -----
    The score rises with the dissolved solute by its definition (zero for the
    dissolved charge, minus one for the reference slurry), so two states that
    give a sensitivity that is not positive are inconsistent.
    """
    ratios, scores = np.asarray(ratios, dtype=float), np.asarray(scores, dtype=float)
    if ratios.shape != (2,) or scores.shape != (2,):
        raise ValueError("Exactly two states are needed; got %d ratios and %d scores." % (ratios.size, scores.size))
    if scores[0] == scores[1]:
        raise ValueError("The two states have the same score (%g)." % scores[0])
    sensitivity = float((ratios[1] - ratios[0]) / (scores[1] - scores[0]))  # [kg/kg]
    if not sensitivity > 0.0:
        raise ValueError("The two states give a sensitivity of %g kg/kg per score unit; the state with more "
                         "dissolved solute must have the higher score." % sensitivity)
    return sensitivity


def ratio_from_score(score, charge: float, sensitivity: float):
    """Convert a score to dissolved solute per solvent.

    Parameters
    ----------
    score : float or numpy.ndarray
        Score of the spectra of one run [-]; zero for its dissolved charge.
    charge : float
        Dissolved solute per solvent of that run with its whole charge dissolved
        [kg/kg].
    sensitivity : float
        Dissolved solute per solvent per unit of score [kg/kg]; positive.

    Returns
    -------
    numpy.ndarray
        Dissolved solute per solvent [kg/kg], shape of ``score``.

    Raises
    ------
    ValueError
        If the sensitivity is not positive.
    """
    return charge + _checked_sensitivity(sensitivity) * np.asarray(score, dtype=float)


def score_from_ratio(ratio, charge: float, sensitivity: float):
    """Predict the score of a solution.

    Parameters
    ----------
    ratio : float or numpy.ndarray
        Dissolved solute per solvent [kg/kg].
    charge : float
        Dissolved solute per solvent of the run with its whole charge dissolved
        [kg/kg].
    sensitivity : float
        Dissolved solute per solvent per unit of score [kg/kg]; positive.

    Returns
    -------
    numpy.ndarray
        Score [-], shape of ``ratio``.

    Raises
    ------
    ValueError
        If the sensitivity is not positive.
    """
    return (np.asarray(ratio, dtype=float) - charge) / _checked_sensitivity(sensitivity)


def charge_from_score(ratio: float, score: float, sensitivity: float) -> float:
    """Return the charge of a run from one state of known composition.

    Parameters
    ----------
    ratio : float
        Dissolved solute per solvent of the known state [kg/kg].
    score : float
        Score of that state [-], counted from the dissolved charge of the run.
    sensitivity : float
        Dissolved solute per solvent per unit of score [kg/kg]; positive.

    Returns
    -------
    float
        Dissolved solute per solvent of the run with its whole charge dissolved
        [kg/kg].

    Raises
    ------
    ValueError
        If the sensitivity is not positive.
    """
    return float(ratio - _checked_sensitivity(sensitivity) * score)
