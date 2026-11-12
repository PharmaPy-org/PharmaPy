"""Height of a spectral band above a straight baseline."""
from __future__ import annotations

import numpy as np


def _nearest_index(axis: np.ndarray, position: float) -> int:
    """Return the index of the axis point closest to ``position``.

    Parameters
    ----------
    axis : numpy.ndarray
        Spectral axis [cm-1], shape ``(n_points,)``, increasing or decreasing.
    position : float
        Requested position [cm-1].

    Returns
    -------
    int
        Index of the nearest point.

    Raises
    ------
    ValueError
        If ``position`` lies outside the axis.
    """
    if not axis.min() <= position <= axis.max():
        raise ValueError("Position %g cm-1 is outside the spectral axis (%g to %g cm-1)." % (position, axis.min(), axis.max()))
    return int(np.argmin(np.abs(axis - position)))


def band_height(axis: np.ndarray, spectra: np.ndarray, peak: float, baseline: tuple[float, float]) -> np.ndarray:
    """Return the height of a band above a straight line through two anchor points.

    Parameters
    ----------
    axis : numpy.ndarray
        Spectral axis [cm-1], shape ``(n_points,)``, increasing or decreasing.
    spectra : numpy.ndarray
        Ordinates, for example absorbance [-], shape ``(n_spectra, n_points)`` or
        ``(n_points,)``.
    peak : float
        Position of the band [cm-1]. The nearest axis point is used.
    baseline : tuple of (float, float)
        Positions of the two baseline anchors [cm-1], at distinct axis points. They
        may lie on the same side of the peak; the line is then extrapolated.

    Returns
    -------
    numpy.ndarray
        Band height in the unit of ``spectra``, shape ``(n_spectra,)``, or a scalar
        array for a single spectrum.

    Raises
    ------
    ValueError
        If a position is outside the axis, the anchors fall on the same axis point,
        or the shapes disagree.
    """
    axis = np.asarray(axis, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    if axis.ndim != 1 or spectra.shape[-1] != axis.size:
        raise ValueError("The last axis of spectra (%s) must match the spectral axis (%s)." % (spectra.shape, axis.shape))
    at_peak = _nearest_index(axis, peak)
    first, second = (_nearest_index(axis, position) for position in baseline)
    if first == second:
        raise ValueError("The baseline anchors %s cm-1 fall on the same axis point." % (baseline,))
    fraction = (axis[at_peak] - axis[first]) / (axis[second] - axis[first])  # [-]
    line = spectra[..., first] + fraction * (spectra[..., second] - spectra[..., first])
    return spectra[..., at_peak] - line
