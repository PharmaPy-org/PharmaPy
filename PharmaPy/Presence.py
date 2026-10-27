"""Structural species presence for solver-based unit operations.

A solver-based unit carries every species as a state, and solver iterates,
difference-quotient Jacobians and round-off give an absent species small
nonzero fractions. Property kernels decide which species need data from
the nonzero entries of their weights (see
``PharmaPy.ThermoModule.ThermoPhysicalManager``), so a unit that passed its
instantaneous fractions would require data for species that can never be
present. ``SpeciesPresence`` instead decides presence structurally, from
the unit's inputs, never from its states (issue #425).
"""

from typing import Optional, Union

import numpy as np


class SpeciesPresence:
    """Latched record of the species that can be present in a unit.

    A species is present if it is in the initial holdup or in any feed or
    composition the unit has recorded, or if the unit marks it explicitly
    (for example a species that a reaction produces, or an internal gas).
    Presence only grows: once recorded, a species stays present until the
    record is rebuilt, so a species whose feed stops keeps needing its
    data while it can remain in the unit. Exact zeros are absent; any other
    value, however small, is present, so the record must be fed only with
    exact inputs (charges and feeds), never with solver states.

    The record also keeps, per species, the earliest absolute time at which
    it was recorded (a charge counts from the start), so that the presence
    that applied at a time can be recovered (``mask_at``): a unit that
    evaluates its balances at time ``t`` needs data only for the species
    recorded at or before ``t``.

    Parameters
    ----------
    num_species : int
        Number of species, in the unit's species order [-].
    """

    def __init__(self, num_species: int) -> None:
        """Start a record in which no species is present.

        Parameters
        ----------
        num_species : int
            Number of species, in the unit's species order [-].
        """
        self._present = np.zeros(int(num_species), dtype=bool)
        # [s], earliest absolute time each species was recorded; -inf for
        # a charge (present from the start), +inf while absent
        self._since = np.full(int(num_species), np.inf)

    def record(self, *amounts: np.ndarray,
               time: Optional[Union[float, np.ndarray]] = None
               ) -> np.ndarray:
        """Mark the species with a nonzero entry in the given amounts.

        Parameters
        ----------
        *amounts : array-like
            Fractions [-], amounts or flows [any basis], or boolean flags,
            shape ``(num_species,)`` or ``(num_rows, num_species)``, in the
            unit's species order. A species with a nonzero entry in any row
            of any argument becomes present.
        time : float or array-like, optional
            Keyword only. Absolute time [s] at which the amounts apply: one
            time for every row, or one per row, shape ``(num_rows,)``. None
            (a charge) means from the start. A species keeps the earliest
            time at which it was recorded.

        Returns
        -------
        numpy.ndarray of bool
            The updated presence mask, shape ``(num_species,)``; a copy.

        Raises
        ------
        ValueError
            If an argument does not have ``num_species`` entries on its last
            axis, or a per-row ``time`` does not have one entry per row.
        """
        num_species = len(self._present)
        for amount in amounts:
            nonzero = np.atleast_2d(np.asarray(amount) != 0)  # [-]
            if nonzero.ndim != 2 or nonzero.shape[-1] != num_species:
                raise ValueError(
                    "Presence amounts must have one entry per species "
                    f"({num_species}) on their last axis; got shape "
                    f"{np.shape(amount)}")
            num_rows = nonzero.shape[0]
            if time is None:
                row_times = np.full(num_rows, -np.inf)  # [s], from the start
            elif np.ndim(time) == 0:
                row_times = np.full(num_rows, float(time))  # [s]
            else:
                row_times = np.asarray(time, dtype=float).ravel()  # [s]
                if row_times.shape != (num_rows,):
                    raise ValueError(
                        "time must be a scalar or have one entry per row "
                        f"({num_rows}); got shape {np.shape(time)}")
            first = np.where(nonzero, row_times[:, np.newaxis],
                             np.inf).min(axis=0)  # [s]
            self._since = np.minimum(self._since, first)  # [s]
            self._present |= nonzero.any(axis=0)

        return self._present.copy()

    def mask_at(self, time: Union[float, np.ndarray]) -> np.ndarray:
        """Return the presence that applied at some times.

        Parameters
        ----------
        time : float or array-like
            Absolute time [s], scalar or shape ``(num_times,)``.

        Returns
        -------
        numpy.ndarray of bool
            True for a species recorded at or before the time, shape
            ``(num_species,)`` for a scalar time and ``(num_times,
            num_species)`` for several.
        """
        time = np.asarray(time, dtype=float)  # [s]
        if time.ndim == 0:
            return self._since <= time

        return self._since[np.newaxis, :] <= time.ravel()[:, np.newaxis]

    @property
    def mask(self) -> np.ndarray:
        """Presence mask, shape ``(num_species,)``; a copy.

        Returns
        -------
        numpy.ndarray of bool
            True for a present species.
        """
        return self._present.copy()

    def weights(self, fractions: np.ndarray,
                used: Optional[np.ndarray] = None,
                time: Optional[Union[float, np.ndarray]] = None
                ) -> np.ndarray:
        """Return presence weights with the shape of some fractions.

        Parameters
        ----------
        fractions : array-like
            Fractions [-] whose shape the weights take, ``(num_species,)``
            or ``(num_rows, num_species)``; their values are not used.
        used : array-like of bool, optional
            Further restriction, shape ``(num_species,)``: only species that
            are also True here get a nonzero weight (for example the
            condensable species of an evaporator).
        time : float or array-like, optional
            Absolute time [s] of the fractions, scalar or one per row,
            shape ``(num_rows,)``: only species recorded at or before it
            count (``mask_at``). None counts every recorded species.

        Returns
        -------
        numpy.ndarray
            1.0 for a present (and used) species and 0.0 otherwise [-],
            broadcast to the shape of ``fractions``, for the ``weights``
            argument of the property kernels.
        """
        present = self._present if time is None else self.mask_at(time)
        if used is not None:
            present = present & np.asarray(used, dtype=bool)
        return np.broadcast_to(present.astype(float),
                               np.shape(fractions)).copy()

    def project(self, fractions: np.ndarray,
                time: Optional[Union[float, np.ndarray]] = None
                ) -> np.ndarray:
        """Set the fractions of absent species to exactly zero.

        Parameters
        ----------
        fractions : array-like
            Fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``, typically solver states.
        time : float or array-like, optional
            Absolute time [s] of the fractions, scalar or one per row: a
            species counts as absent before it was recorded (``mask_at``).
            None uses every recorded species.

        Returns
        -------
        numpy.ndarray
            A copy with the entries of absent species replaced by 0.0 and
            the others unchanged, not renormalized; the removed entries are
            round-off of an exact zero.
        """
        present = self._present if time is None else self.mask_at(time)
        return np.where(present, np.asarray(fractions, dtype=float), 0.0)
