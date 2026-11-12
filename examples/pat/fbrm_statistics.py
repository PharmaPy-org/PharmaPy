"""Statistics of chord-length distributions as iC FBRM reports them.

Definitions were checked against the statistics the vendor software stored for the
caffeine run of 2026-10-23: unweighted counts per band and the square-weighted
mean chord on the 100-channel logarithmic grid from 1 to 1000 um.
"""
from __future__ import annotations

import numpy as np

from .ic_fbrm import rebin_matrix


def log_channel_edges(lower: float = 1.0, upper: float = 1000.0, count: int = 100) -> np.ndarray:
    """Return logarithmically spaced channel edges.

    Parameters
    ----------
    lower, upper : float, optional
        First and last edge [um]. Defaults 1 and 1000 um, the grid of the iC FBRM
        export.
    count : int, optional
        Number of channels [-]. Default 100.

    Returns
    -------
    numpy.ndarray
        Edges [um], shape ``(count + 1,)``.

    Raises
    ------
    ValueError
        If the limits are not positive and increasing or ``count`` is below 1.
    """
    if not 0.0 < lower < upper or count < 1:
        raise ValueError("log_channel_edges needs 0 < lower < upper and count >= 1.")
    return lower * (upper / lower) ** (np.arange(count + 1) / count)


def geometric_midpoints(edges: np.ndarray) -> np.ndarray:
    """Return the geometric mean of each pair of neighbouring edges.

    Parameters
    ----------
    edges : numpy.ndarray
        Positive channel edges [um], shape ``(n_channels + 1,)``.

    Returns
    -------
    numpy.ndarray
        Channel midpoints [um], shape ``(n_channels,)``; the midpoints the iC FBRM
        export prints for its logarithmic grid.
    """
    edges = np.asarray(edges, dtype=float)
    return np.sqrt(edges[:-1] * edges[1:])


def band_counts(counts: np.ndarray, channel_edges: np.ndarray, band_edges: np.ndarray) -> np.ndarray:
    """Sum unweighted counts into chord-length bands.

    Parameters
    ----------
    counts : numpy.ndarray
        Counts per channel [counts], shape ``(n_samples, n_channels)``.
    channel_edges : numpy.ndarray
        Edges of those channels [um], shape ``(n_channels + 1,)``.
    band_edges : numpy.ndarray
        Edges of the bands [um], shape ``(n_bands + 1,)``, increasing.

    Returns
    -------
    numpy.ndarray
        Counts per band [counts], shape ``(n_samples, n_bands)``. A channel that
        straddles a band edge is split by overlapping length.
    """
    return np.asarray(counts, dtype=float) @ rebin_matrix(channel_edges, band_edges)


def weighted_mean_chord(counts: np.ndarray, midpoints: np.ndarray, weight_power: float = 2.0) -> np.ndarray:
    """Return the length-weighted mean chord of each sample.

    Parameters
    ----------
    counts : numpy.ndarray
        Counts per channel [counts], shape ``(n_samples, n_channels)``.
    midpoints : numpy.ndarray
        Channel midpoints [um], shape ``(n_channels,)``.
    weight_power : float, optional
        Power ``p`` of the chord length used as weight [-]. Default 2, the
        "square weight" of iC FBRM; 0 gives the unweighted mean.

    Returns
    -------
    numpy.ndarray
        ``sum(n L**(p + 1)) / sum(n L**p)`` [um], shape ``(n_samples,)``. NaN for a
        sample without counts.
    """
    counts = np.asarray(counts, dtype=float)
    weights = counts * np.asarray(midpoints, dtype=float) ** weight_power  # [counts um**p]
    total = weights.sum(axis=1)
    mean = np.full(total.shape, np.nan)  # [um]
    filled = total > 0
    mean[filled] = (weights[filled] * midpoints).sum(axis=1) / total[filled]
    return mean
