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

from typing import Optional

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

    def record(self, *amounts: np.ndarray) -> np.ndarray:
        """Mark the species with a nonzero entry in the given amounts.

        Parameters
        ----------
        *amounts : array-like
            Fractions [-], amounts or flows [any basis], or boolean flags,
            shape ``(num_species,)`` or ``(num_rows, num_species)``, in the
            unit's species order. A species with a nonzero entry in any row
            of any argument becomes present.

        Returns
        -------
        numpy.ndarray of bool
            The updated presence mask, shape ``(num_species,)``; a copy.

        Raises
        ------
        ValueError
            If an argument does not have ``num_species`` entries on its last
            axis.
        """
        num_species = len(self._present)
        for amount in amounts:
            nonzero = np.atleast_2d(np.asarray(amount) != 0)  # [-]
            if nonzero.ndim != 2 or nonzero.shape[-1] != num_species:
                raise ValueError(
                    "Presence amounts must have one entry per species "
                    f"({num_species}) on their last axis; got shape "
                    f"{np.shape(amount)}")
            self._present |= nonzero.any(axis=0)

        return self._present.copy()

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
                used: Optional[np.ndarray] = None) -> np.ndarray:
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

        Returns
        -------
        numpy.ndarray
            1.0 for a present (and used) species and 0.0 otherwise [-],
            broadcast to the shape of ``fractions``, for the ``weights``
            argument of the property kernels.
        """
        present = self._present if used is None else (
            self._present & np.asarray(used, dtype=bool))
        return np.broadcast_to(present.astype(float),
                               np.shape(fractions)).copy()

    def project(self, fractions: np.ndarray) -> np.ndarray:
        """Set the fractions of absent species to exactly zero.

        Parameters
        ----------
        fractions : array-like
            Fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``, typically solver states.

        Returns
        -------
        numpy.ndarray
            A copy with the entries of absent species replaced by 0.0 and
            the others unchanged, not renormalized; the removed entries are
            round-off of an exact zero.
        """
        return np.where(self._present, np.asarray(fractions, dtype=float),
                        0.0)
