"""Signals of the dissolved solute from spectra of a process-analytical-technology probe.

`band_height` reduces a spectrum to the height of one band above a straight
baseline, for a solute with a band the solvent leaves free. `SpectralScore` uses
the whole spectrum, for a solute without such a band: it is fitted on a run with
stretches of known state and then applied to the spectra of any run. Neither
signal is a concentration; a calibration against known compositions gives one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COMPONENT_COUNT = 4  # direction, reference, offset, tilt


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


def score_terms(axis: np.ndarray, spectra: np.ndarray, reference: np.ndarray, direction: np.ndarray) -> ScoreTerms:
    """Split the difference between spectra and a reference into score and contact terms.

    The difference is fitted by least squares as::

        difference = score * direction + scale * reference + offset + tilt * position

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
        One value of every term per spectrum.

    Raises
    ------
    ValueError
        If the shapes disagree, or the direction cannot be separated from the
        reference, a constant and a line.

    Notes
    -----
    ``scale``, ``offset`` and ``tilt`` take up a change of the contact between
    probe and liquid, which multiplies the whole spectrum and shifts and tilts
    its baseline. The multiplication reaches the solute change too: a spectrum
    ``(1 + g) * (reference + s * direction)`` gives the scale ``g`` and the score
    ``s * (1 + g)``. The score is therefore off by the fraction ``g`` of itself,
    which the scale term reports.
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


def _selection(mask: np.ndarray, count: int, what: str) -> np.ndarray:
    """Return a selection of spectra after checking it.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask of the spectra selected, shape ``(n_spectra,)``.
    count : int
        Number of spectra [-].
    what : str
        Name of the selection, for the error message.

    Returns
    -------
    numpy.ndarray
        The mask as a boolean array.

    Raises
    ------
    ValueError
        If the mask is not boolean, has another shape, or selects no spectrum.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (count,):
        raise ValueError("The %s selection must be a boolean mask with one entry per spectrum (%d); got %s of shape %s."
                         % (what, count, mask.dtype, mask.shape))
    if not mask.any():
        raise ValueError("The %s selection holds no spectrum." % what)
    return mask


@dataclass(frozen=True)
class SpectralScore:
    """Whole-spectrum score of the dissolved solute, fitted on a reference run.

    `fit` learns two things from one run with stretches of known state: how
    temperature alone changes the spectrum, and how the spectrum changes between
    a slurry and the dissolved charge (the direction). `score` then reads any
    spectrum of the same solute, solvent and probe in units of that change. The
    temperature effect is taken as linear in temperature at every spectral
    point; outside ``temperature_range`` it is an extrapolation.

    Parameters
    ----------
    wavenumber : numpy.ndarray
        Spectral axis of the spectra [cm-1], shape ``(n_points,)``.
    usable : numpy.ndarray
        Boolean mask of the spectral points used, shape ``(n_points,)``.
    per_kelvin : numpy.ndarray
        Change of the absorbance with temperature [1/K], shape ``(n_points,)``.
    reference_temperature : float
        Temperature every spectrum is moved to before it is read [K].
    direction : numpy.ndarray
        Dissolved minus slurry spectrum of the reference run at the reference
        temperature [-] (absorbance), shape ``(n_usable,)``.
    temperature_range : tuple of (float, float)
        Lowest and highest temperature of the clear-solution spectra [K].
    temperature_residual : float
        Root-mean-square difference between the clear-solution spectra and their
        lines in temperature [-] (absorbance).
    score_per_kelvin : float
        Score that the temperature effect of one kelvin would read as if it were
        not removed [1/K]. Times the distance of a spectrum from
        ``temperature_range`` [K], it is the part of that spectrum's score that
        rests on the extrapolation.
    """

    wavenumber: np.ndarray
    usable: np.ndarray
    per_kelvin: np.ndarray
    reference_temperature: float
    direction: np.ndarray
    temperature_range: tuple[float, float]
    temperature_residual: float
    score_per_kelvin: float

    @classmethod
    def fit(cls, wavenumber: np.ndarray, spectra: np.ndarray, temperature: np.ndarray, *, clear_solution: np.ndarray,
            dissolved: np.ndarray, slurry: np.ndarray, reference_temperature: float, window: tuple[float, float],
            maximum_absorbance: float) -> SpectralScore:
        """Fit the score on the spectra of a reference run.

        Parameters
        ----------
        wavenumber : numpy.ndarray
            Spectral axis [cm-1], shape ``(n_points,)``.
        spectra : numpy.ndarray
            Absorbance of the reference run [-], shape ``(n_spectra, n_points)``.
        temperature : numpy.ndarray
            Liquid temperature of each spectrum [K], shape ``(n_spectra,)``.
        clear_solution : numpy.ndarray
            Boolean mask, shape ``(n_spectra,)``, of spectra of one composition
            without solid at two or more temperatures. They give the temperature
            effect.
        dissolved : numpy.ndarray
            Boolean mask, shape ``(n_spectra,)``, of spectra with the whole charge
            dissolved. They read score 0.
        slurry : numpy.ndarray
            Boolean mask, shape ``(n_spectra,)``, of spectra of a slurry of known
            state. They read score -1.
        reference_temperature : float
            Temperature the spectra are moved to [K].
        window : tuple of (float, float)
            Lower and upper limit of the spectral window used [cm-1].
        maximum_absorbance : float
            Largest absorbance of the clear solution kept [-]; points on stronger
            bands are left out.

        Returns
        -------
        SpectralScore
            The fitted score.

        Raises
        ------
        ValueError
            If the shapes disagree, a selection is not a boolean mask or is
            empty, the clear solution spans one temperature, no spectral point is
            usable, or the direction cannot be told from a contact change.
        """
        axis, spectra = np.asarray(wavenumber, dtype=float), np.asarray(spectra, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        if axis.ndim != 1 or spectra.ndim != 2 or spectra.shape[1] != axis.size or temperature.shape != (spectra.shape[0],):
            raise ValueError("Shapes disagree: axis %s, spectra %s, temperature %s." % (axis.shape, spectra.shape, temperature.shape))
        clear, zero, minus_one = (_selection(mask, spectra.shape[0], what) for mask, what in (
            (clear_solution, "clear-solution"), (dissolved, "dissolved"), (slurry, "slurry")))
        level, per_kelvin, residual = temperature_effect(spectra[clear], temperature[clear], reference_temperature)
        usable = usable_points(axis, level, window[0], window[1], maximum_absorbance)
        moved = at_reference_temperature(spectra, temperature, per_kelvin, reference_temperature)[:, usable]
        reference = moved[zero].mean(axis=0)  # [-]
        direction = reference - moved[minus_one].mean(axis=0)  # [-]
        warmer = score_terms(axis[usable], (reference + per_kelvin[usable])[None, :], reference, direction)
        return cls(wavenumber=axis, usable=usable, per_kelvin=per_kelvin, reference_temperature=float(reference_temperature),
                   direction=direction, temperature_range=(float(temperature[clear].min()), float(temperature[clear].max())),
                   temperature_residual=residual, score_per_kelvin=float(warmer.score[0]))

    def at_reference(self, wavenumber: np.ndarray, spectra: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Move spectra to the reference temperature and keep the usable points.

        Parameters
        ----------
        wavenumber : numpy.ndarray
            Spectral axis of ``spectra`` [cm-1], shape ``(n_points,)``. It must be
            the axis the score was fitted on.
        spectra : numpy.ndarray
            Absorbance [-], shape ``(n_spectra, n_points)``.
        temperature : numpy.ndarray
            Liquid temperature of each spectrum [K], shape ``(n_spectra,)``.

        Returns
        -------
        numpy.ndarray
            Absorbance at the reference temperature [-], shape
            ``(n_spectra, n_usable)``, at ``wavenumber[usable]``.

        Raises
        ------
        ValueError
            If the axis differs from the fitted one, the shapes disagree, or a
            temperature is not finite.
        """
        axis = np.asarray(wavenumber, dtype=float)
        if axis.shape != self.wavenumber.shape or not np.array_equal(axis, self.wavenumber):
            raise ValueError("The spectra are not on the spectral axis the score was fitted on.")
        return at_reference_temperature(spectra, temperature, self.per_kelvin, self.reference_temperature)[:, self.usable]

    def score(self, wavenumber: np.ndarray, spectra: np.ndarray, temperature: np.ndarray, dissolved: np.ndarray) -> ScoreTerms:
        """Read the spectra of a run in units of the fitted direction.

        Parameters
        ----------
        wavenumber : numpy.ndarray
            Spectral axis of ``spectra`` [cm-1], shape ``(n_points,)``. It must be
            the axis the score was fitted on.
        spectra : numpy.ndarray
            Absorbance of the run [-], shape ``(n_spectra, n_points)``.
        temperature : numpy.ndarray
            Liquid temperature of each spectrum [K], shape ``(n_spectra,)``.
        dissolved : numpy.ndarray
            Boolean mask, shape ``(n_spectra,)``, of the spectra of this run with
            its whole charge dissolved. Their mean is the zero of the score.

        Returns
        -------
        ScoreTerms
            Score and contact terms of every spectrum. The score is 0 for the
            run's own dissolved charge and -1 for a loss of dissolved solute equal
            to the one between the dissolved charge and the slurry of the
            reference run.

        Raises
        ------
        ValueError
            If the axis differs from the fitted one, the shapes disagree, a
            temperature is not finite, or the selection is not a boolean mask or
            is empty.
        """
        moved = self.at_reference(wavenumber, spectra, temperature)
        reference = moved[_selection(dissolved, moved.shape[0], "dissolved")].mean(axis=0)  # [-]
        return score_terms(self.wavenumber[self.usable], moved, reference, self.direction)
