"""Measurement predictions from the state of a crystallizing slurry.

A crystallizer result is first brought to a `SlurryState`: crystal moments on
a total basis in SI lengths, dissolved solute and solvent masses, and slurry
volume. The predictions are functions of that state only, so they do not
depend on which model produced it. One adapter per model layout builds the
state. This module defines no experiment or measurement objects; those are
``PharmaPy.ParamEstim.Experiment`` and ``Measurement``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PharmaPy.Errors import PharmaPyValueError

METERS_PER_MICROMETER = 1e-6  # [m/um], exact
# The crystal volume is shape_factor * mu_3, so orders 0 to 3 are needed.
MINIMUM_MOMENT_COUNT = 4  # [-]


@dataclass(frozen=True)
class SlurryState:
    """State of a batch slurry at the output times of a model.

    Parameters
    ----------
    time : numpy.ndarray
        Times [s], shape ``(n_times,)``.
    moments : numpy.ndarray
        Moments of the crystal number distribution over the characteristic
        length, total over the vessel: order k in [m**k], shape
        ``(n_times, n_moments)``, orders 0 to ``n_moments - 1``.
    solute_mass : numpy.ndarray
        Dissolved solute [kg], shape ``(n_times,)``.
    solvent_mass : numpy.ndarray
        Solvent in the liquid [kg], shape ``(n_times,)``.
    slurry_volume : numpy.ndarray
        Volume of liquid plus crystals [m**3], shape ``(n_times,)``.
    """

    time: np.ndarray
    moments: np.ndarray
    solute_mass: np.ndarray
    solvent_mass: np.ndarray
    slurry_volume: np.ndarray


def _length_factors(count: int) -> np.ndarray:
    """Return the conversion of moments from micrometers to meters.

    Parameters
    ----------
    count : int
        Number of moments, orders 0 to ``count - 1``.

    Returns
    -------
    numpy.ndarray
        Factor of each order [m**k/um**k], shape ``(count,)``.
    """
    return METERS_PER_MICROMETER ** np.arange(count)


def _check_species(solute_index: int, solvent_index: int, count: int) -> None:
    """Check that the solute and the solvent are two of the species.

    Parameters
    ----------
    solute_index, solvent_index : int
        Positions of the solute and the solvent among the species.
    count : int
        Number of species.

    Raises
    ------
    PharmaPyValueError
        If a position is outside ``0`` to ``count - 1`` or both coincide.
    """
    if (not 0 <= solute_index < count or not 0 <= solvent_index < count
            or solute_index == solvent_index):
        raise PharmaPyValueError(
            f"Solute index {solute_index} and solvent index {solvent_index} "
            f"must be two different positions among {count} species.")


def _check_positive(value: float, name: str) -> None:
    """Check that a scalar physical constant is finite and positive.

    Parameters
    ----------
    value : float
        The constant.
    name : str
        Its name, for the message.

    Raises
    ------
    PharmaPyValueError
        If ``value`` is not a finite positive number.
    """
    if not np.isfinite(value) or value <= 0:
        raise PharmaPyValueError(
            f"{name} must be finite and positive; got {value}.")


def state_from_total_moments(time, states, solute_index: int,
                             solvent_index: int, shape_factor: float,
                             moment_count: int = MINIMUM_MOMENT_COUNT
                             ) -> SlurryState:
    """Build the slurry state from total moments and liquid concentrations.

    This is the state layout of ``PharmaPy.Crystallizers.BatchCryst`` with
    the method of moments, as ``solve_unit`` and ``paramest_wrapper`` return
    it.

    Parameters
    ----------
    time : array-like
        Output times [s], shape ``(n_times,)``.
    states : array-like
        Shape ``(n_times, moment_count + n_species + 1)``: the total moments,
        order k in [um**k]; the liquid mass concentration of every species
        [kg/m**3]; and the liquid volume [m**3].
    solute_index, solvent_index : int
        Positions of the solute and the solvent among the species.
    shape_factor : float
        Crystal volume over the cube of the characteristic length [-].
    moment_count : int, optional
        Number of moments. Default 4.

    Returns
    -------
    SlurryState
        The state; the slurry volume is the liquid volume plus the crystal
        volume ``shape_factor * mu_3``.

    Raises
    ------
    PharmaPyValueError
        If the shapes disagree, fewer than four moments are given, the
        species positions are invalid, or the shape factor is not positive.
    """
    time = np.asarray(time, dtype=float)
    states = np.asarray(states, dtype=float)
    _check_positive(shape_factor, "shape_factor")
    if moment_count < MINIMUM_MOMENT_COUNT:
        raise PharmaPyValueError(
            "At least four moments are needed for the crystal volume; got "
            f"{moment_count}.")
    if (time.ndim != 1 or states.ndim != 2 or states.shape[0] != time.size
            or states.shape[1] < moment_count + 3):
        raise PharmaPyValueError(
            f"States of shape {states.shape} do not hold {moment_count} "
            f"moments, at least two species and a volume for {time.size} "
            "times.")
    moments = states[:, :moment_count] * _length_factors(moment_count)  # [m**k]
    concentration = states[:, moment_count:-1]  # [kg/m**3]
    _check_species(solute_index, solvent_index, concentration.shape[1])
    liquid_volume = states[:, -1]  # [m**3]
    return SlurryState(
        time, moments, concentration[:, solute_index] * liquid_volume,
        concentration[:, solvent_index] * liquid_volume,
        liquid_volume + shape_factor * moments[:, 3])


def state_from_specific_moments(time, moments_per_volume, species_mass,
                                slurry_volume, solute_index: int,
                                solvent_index: int) -> SlurryState:
    """Build the slurry state from moments per volume and species masses.

    This is the result layout of
    ``PharmaPy.Crystallizers_Refactored.BatchCrystallizer`` with the method
    of moments: ``result.mu_n_solid0``, ``result.mass_j_liquid0`` and
    ``result.vessel_vol``.

    Parameters
    ----------
    time : array-like
        Output times [s], shape ``(n_times,)``.
    moments_per_volume : array-like
        Moments per slurry volume, order k in [um**k/m**3], shape
        ``(n_times, n_moments)``.
    species_mass : array-like
        Mass of every species in the liquid [kg], shape
        ``(n_times, n_species)``.
    slurry_volume : array-like
        Volume of liquid plus crystals [m**3], shape ``(n_times,)``.
    solute_index, solvent_index : int
        Positions of the solute and the solvent among the species.

    Returns
    -------
    SlurryState
        The state, with total moments in SI lengths.

    Raises
    ------
    PharmaPyValueError
        If the shapes disagree or the species positions are invalid.
    """
    time = np.asarray(time, dtype=float)
    moments_per_volume = np.asarray(moments_per_volume, dtype=float)
    species_mass = np.asarray(species_mass, dtype=float)
    slurry_volume = np.asarray(slurry_volume, dtype=float)
    if (time.ndim != 1 or moments_per_volume.ndim != 2
            or species_mass.ndim != 2 or slurry_volume.shape != time.shape
            or moments_per_volume.shape[0] != time.size
            or species_mass.shape[0] != time.size):
        raise PharmaPyValueError(
            f"Moments {moments_per_volume.shape}, species masses "
            f"{species_mass.shape} and volume {slurry_volume.shape} do not "
            f"match {time.size} times.")
    _check_species(solute_index, solvent_index, species_mass.shape[1])
    moments = (moments_per_volume
               * _length_factors(moments_per_volume.shape[1])
               * slurry_volume[:, np.newaxis])  # [m**k]
    return SlurryState(time, moments, species_mass[:, solute_index],
                       species_mass[:, solvent_index], slurry_volume)


def dissolved_ratio(state: SlurryState) -> np.ndarray:
    """Return the dissolved solute per solvent.

    Parameters
    ----------
    state : SlurryState
        Slurry state.

    Returns
    -------
    numpy.ndarray
        Dissolved solute per solvent [kg/kg], shape ``(n_times,)``.
    """
    return state.solute_mass / state.solvent_mass


def solid_mass(state: SlurryState, solute_density: float,
               shape_factor: float) -> np.ndarray:
    """Return the solute held in the crystals.

    Parameters
    ----------
    state : SlurryState
        Slurry state with at least four moments.
    solute_density : float
        Mass of solute per crystal volume [kg/m**3]. For a solvate it is the
        solute content of the crystal, not the crystal density.
    shape_factor : float
        Crystal volume over the cube of the characteristic length [-].

    Returns
    -------
    numpy.ndarray
        Solute in the solid [kg], shape ``(n_times,)``:
        ``solute_density * shape_factor * mu_3``.

    Raises
    ------
    PharmaPyValueError
        If the density or the shape factor is not positive, or the state
        holds fewer than four moments.
    """
    _check_positive(solute_density, "solute_density")
    _check_positive(shape_factor, "shape_factor")
    if state.moments.shape[1] < MINIMUM_MOMENT_COUNT:
        raise PharmaPyValueError(
            "The solid mass needs the third moment; the state holds "
            f"{state.moments.shape[1]} moments.")
    return solute_density * shape_factor * state.moments[:, 3]


def total_solute(state: SlurryState, solute_density: float,
                 shape_factor: float) -> np.ndarray:
    """Return the solute in the vessel, dissolved plus solid.

    Parameters
    ----------
    state : SlurryState
        Slurry state with at least four moments.
    solute_density, shape_factor : float
        As in `solid_mass`.

    Returns
    -------
    numpy.ndarray
        Solute in the vessel [kg], shape ``(n_times,)``. In a closed batch it
        stays at the charge plus the seed; its drift measures how well the
        model conserves the solute.
    """
    return state.solute_mass + solid_mass(state, solute_density, shape_factor)


def count_signal(state: SlurryState, order: int,
                 constant: float) -> np.ndarray:
    """Predict a particle-count signal from a moment per slurry volume.

    Parameters
    ----------
    state : SlurryState
        Slurry state.
    order : int
        Moment order [-]; 0 counts crystals, 1 weights them by length, 2 by
        area.
    constant : float
        Instrument constant [counts * m**(3 - order)]. It absorbs the optics
        and the crystal shape and has to be fitted.

    Returns
    -------
    numpy.ndarray
        Predicted counts [counts], shape ``(n_times,)``:
        ``constant * mu_order / slurry_volume``.

    Raises
    ------
    PharmaPyValueError
        If the state does not hold a moment of that order.

    Notes
    -----
    A modeling assumption, not an instrument model: it holds only while the
    crystals keep their shape, so that every chord scales with the
    characteristic length.
    """
    if not 0 <= order < state.moments.shape[1]:
        raise PharmaPyValueError(
            f"The state holds moments 0 to {state.moments.shape[1] - 1}; "
            f"order {order} is not among them.")
    return constant * state.moments[:, order] / state.slurry_volume


def mean_size(state: SlurryState, upper: int = 3,
              lower: int = 2) -> np.ndarray:
    """Return a mean crystal size as the ratio of two consecutive moments.

    Parameters
    ----------
    state : SlurryState
        Slurry state.
    upper, lower : int, optional
        Moment orders with ``upper == lower + 1``. The default 3 and 2 gives
        the Sauter mean.

    Returns
    -------
    numpy.ndarray
        Mean of the characteristic length [m], shape ``(n_times,)``; NaN
        where the lower moment is zero.

    Raises
    ------
    PharmaPyValueError
        If the orders are not consecutive or not held by the state.
    """
    if upper != lower + 1 or lower < 0 or upper >= state.moments.shape[1]:
        raise PharmaPyValueError(
            f"Orders {upper} and {lower} are not consecutive moments of the "
            "state.")
    denominator = state.moments[:, lower]
    return np.divide(state.moments[:, upper], denominator,
                     out=np.full(denominator.shape, np.nan),
                     where=denominator > 0.0)
