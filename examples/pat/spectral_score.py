"""Dissolved-solute score of whole spectra, relative to a reference spectrum.

A spectrum is first moved to a reference temperature with a linear temperature
effect per spectral point. Its difference from a reference spectrum is then split
into four terms::

    difference = score * direction + scale * reference + offset + tilt * position

``direction`` is the change of the spectrum for a known change of the dissolved
solute, so ``score`` counts that change. The other three terms take up a change of
the contact between probe and liquid, which multiplies the whole spectrum and
shifts and tilts its baseline. Nothing here is specific to one compound: the
spectra that define the temperature effect, the reference and the direction are
chosen by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COMPONENT_COUNT = 4  # direction, reference, offset, tilt


@dataclass(frozen=True)
class ScoreTerms:
    """Terms of the decomposition of spectra against a reference.

    Parameters
    ----------
    score : numpy.ndarray
        Multiples of ``direction`` [-], shape ``(n_spectra,)``.
    scale : numpy.ndarray
        Multiples of the reference spectrum [-], shape ``(n_spectra,)``.
    offset : numpy.ndarray
        Constant baseline shift [-] (absorbance), shape ``(n_spectra,)``.
    tilt : numpy.ndarray
        Baseline change across the whole spectral range [-] (absorbance), shape
        ``(n_spectra,)``.
    residual : numpy.ndarray
        Root-mean-square of what the four terms leave unexplained [-]
        (absorbance), shape ``(n_spectra,)``.
    condition : float
        Condition number of the four components scaled to unit length [-]. A
        large value means the direction is hard to tell from a contact change.
    """

    score: np.ndarray
    scale: np.ndarray
    offset: np.ndarray
    tilt: np.ndarray
    residual: np.ndarray
    condition: float


def usable_points(axis: np.ndarray, level: np.ndarray, lower: float, upper: float, maximum: float) -> np.ndarray:
    """Return the spectral points inside a window whose absorbance stays below a limit.

    Parameters
    ----------
    axis : numpy.ndarray
        Spectral axis [cm-1], shape ``(n_points,)``.
    level : numpy.ndarray
        Spectrum that shows the strongest bands, for example the dissolved
        reference [-] (absorbance), shape ``(n_points,)``.
    lower, upper : float
        Limits of the window [cm-1], both included.
    maximum : float
        Largest absolute absorbance kept [-]. Points above it lie on bands too
        strong for a linear response.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shape ``(n_points,)``.

    Raises
    ------
    ValueError
        If the shapes disagree, the window is empty or reversed, or no point is left.
    """
    axis, level = np.asarray(axis, dtype=float), np.asarray(level, dtype=float)
    if axis.ndim != 1 or level.shape != axis.shape:
        raise ValueError("The level spectrum (%s) must match the spectral axis (%s)." % (level.shape, axis.shape))
    if not lower < upper:
        raise ValueError("The window limits must increase; got %g to %g cm-1." % (lower, upper))
    keep = (axis >= lower) & (axis <= upper) & (np.abs(level) <= maximum)
    if not keep.any():
        raise ValueError("No spectral point between %g and %g cm-1 has an absorbance of at most %g." % (lower, upper, maximum))
    return keep


def temperature_effect(spectra: np.ndarray, temperature: np.ndarray, reference_temperature: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a straight line in temperature to every spectral point.

    Parameters
    ----------
    spectra : numpy.ndarray
        Spectra of one composition at several temperatures [-] (absorbance), shape
        ``(n_spectra, n_points)``. A clear solution of constant concentration.
    temperature : numpy.ndarray
        Temperature of each spectrum [K] or [C], shape ``(n_spectra,)``.
    reference_temperature : float
        Temperature the line is centred on, in the unit of ``temperature``.

    Returns
    -------
    at_reference : numpy.ndarray
        Spectrum at the reference temperature [-], shape ``(n_points,)``.
    per_kelvin : numpy.ndarray
        Change of the absorbance with temperature [1/K], shape ``(n_points,)``.
    residual : float
        Root-mean-square difference between the spectra and the lines [-].

    Raises
    ------
    ValueError
        If the shapes disagree or the spectra do not span two temperatures.
    """
    spectra, temperature = np.asarray(spectra, dtype=float), np.asarray(temperature, dtype=float)
    if spectra.ndim != 2 or temperature.shape != (spectra.shape[0],):
        raise ValueError("Expected spectra of shape (n_spectra, n_points) and one temperature each; got %s and %s."
                         % (spectra.shape, temperature.shape))
    if np.ptp(temperature) <= 0.0:
        raise ValueError("The spectra must span at least two temperatures to give a temperature effect.")
    design = np.column_stack((np.ones(temperature.size), temperature - reference_temperature))
    coefficients, *_ = np.linalg.lstsq(design, spectra, rcond=None)
    residual = float(np.sqrt(np.mean((spectra - design @ coefficients) ** 2)))  # [-]
    return coefficients[0], coefficients[1], residual


def at_reference_temperature(spectra: np.ndarray, temperature: np.ndarray, per_kelvin: np.ndarray,
                             reference_temperature: float) -> np.ndarray:
    """Move spectra to the reference temperature with the linear temperature effect.

    Parameters
    ----------
    spectra : numpy.ndarray
        Spectra [-] (absorbance), shape ``(n_spectra, n_points)``.
    temperature : numpy.ndarray
        Temperature of each spectrum, shape ``(n_spectra,)``, in the unit used for
        ``per_kelvin``.
    per_kelvin : numpy.ndarray
        Change of the absorbance with temperature [1/K], shape ``(n_points,)``.
    reference_temperature : float
        Reference temperature, in the unit of ``temperature``.

    Returns
    -------
    numpy.ndarray
        The spectra as they would read at the reference temperature [-], shape
        ``(n_spectra, n_points)``.

    Raises
    ------
    ValueError
        If the shapes disagree or a temperature is not finite.
    """
    spectra, temperature = np.asarray(spectra, dtype=float), np.asarray(temperature, dtype=float)
    per_kelvin = np.asarray(per_kelvin, dtype=float)
    if spectra.ndim != 2 or temperature.shape != (spectra.shape[0],) or per_kelvin.shape != (spectra.shape[1],):
        raise ValueError("Shapes disagree: spectra %s, temperature %s, temperature effect %s."
                         % (spectra.shape, temperature.shape, per_kelvin.shape))
    if not np.all(np.isfinite(temperature)):
        raise ValueError("Every spectrum needs a finite temperature.")
    return spectra - np.outer(temperature - reference_temperature, per_kelvin)


def score(axis: np.ndarray, spectra: np.ndarray, reference: np.ndarray, direction: np.ndarray) -> ScoreTerms:
    """Split the difference between spectra and a reference into score and contact terms.

    Parameters
    ----------
    axis : numpy.ndarray
        Spectral axis [cm-1], shape ``(n_points,)``. It gives the tilt term its
        shape: a line from -1/2 to +1/2 across the range.
    spectra : numpy.ndarray
        Spectra at the reference temperature [-] (absorbance), shape
        ``(n_spectra, n_points)``.
    reference : numpy.ndarray
        Reference spectrum at the same temperature [-], shape ``(n_points,)``.
    direction : numpy.ndarray
        Change of the spectrum for the known change of dissolved solute [-], shape
        ``(n_points,)``.

    Returns
    -------
    ScoreTerms
        One value of every term per spectrum, from a least-squares fit.

    Raises
    ------
    ValueError
        If the shapes disagree, or the direction cannot be separated from the
        reference, a constant and a line.

    Notes
    -----
    A change of probe contact multiplies the whole spectrum, the solute change
    included: a spectrum ``(1 + g) * (reference + s * direction)`` gives the scale
    ``g`` and the score ``s * (1 + g)``. The score is therefore off by the
    fraction ``g`` of itself, which the scale term reports.
    """
    axis, spectra = np.asarray(axis, dtype=float), np.asarray(spectra, dtype=float)
    reference, direction = np.asarray(reference, dtype=float), np.asarray(direction, dtype=float)
    if axis.ndim != 1 or spectra.ndim != 2 or spectra.shape[1] != axis.size or reference.shape != axis.shape or direction.shape != axis.shape:
        raise ValueError("Shapes disagree: axis %s, spectra %s, reference %s, direction %s."
                         % (axis.shape, spectra.shape, reference.shape, direction.shape))
    position = (axis - axis.mean()) / np.ptp(axis)  # [-]
    components = np.column_stack((direction, reference, np.ones(axis.size), position))
    if np.linalg.matrix_rank(components) < COMPONENT_COUNT:
        raise ValueError("The direction cannot be separated from the reference spectrum, a constant and a line.")
    difference = spectra - reference
    amounts, *_ = np.linalg.lstsq(components, difference.T, rcond=None)
    residual = np.sqrt(np.mean((difference - (components @ amounts).T) ** 2, axis=1))  # [-]
    condition = float(np.linalg.cond(components / np.linalg.norm(components, axis=0)))  # [-]
    return ScoreTerms(score=amounts[0], scale=amounts[1], offset=amounts[2], tilt=amounts[3], residual=residual, condition=condition)
