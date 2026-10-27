"""Structural species presence for solver-based unit operations.

A solver-based unit carries every species as a state, and solver iterates,
difference-quotient Jacobians and round-off give an absent species small
nonzero fractions. Property kernels decide which species need data from
the nonzero entries of their weights (see
``PharmaPy.ThermoModule.ThermoPhysicalManager``), so a unit that passed its
instantaneous fractions would require data for species that can never be
present. ``SpeciesPresence`` instead decides presence structurally, from
the unit's inputs, never from its states (issue #425).
``StructuralPresenceMixin`` gives unit operations the shared wiring: the
record's lifecycle, the kernel weights, the feed latches, the projection
of published fractions and the rollback of a failed solve.
"""

import copy
from typing import Optional, Tuple, Union

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


def feed_weights(flow: Union[float, np.ndarray],
                  fractions: np.ndarray) -> np.ndarray:
    """Flag the feed entries that actually flow.

    Parameters
    ----------
    flow : float or array-like
        Effective feed flow [mol/s] or [kg/s], or, for a batch inlet, its
        amount [mol] or [kg], on the basis of ``fractions``, scalar or
        shape ``(num_rows,)``, after any shut-off; only whether it is
        exactly zero matters.
    fractions : array-like
        Feed mole or mass fractions [-], shape ``(num_species,)`` or
        ``(num_rows, num_species)``.

    Returns
    -------
    numpy.ndarray
        1.0 where both the row's flow and the species' fraction are
        exactly nonzero, 0.0 otherwise [-]: shape ``(num_species,)``
        for a scalar flow, ``(num_rows, num_species)`` for a flow
        profile, and ``(1, num_species)`` for one composition row paired
        with a flow profile.

    Notes
    -----
    Units that keep no presence record (``Mixer``, the liquid
    ``DynamicCollector``) pass these weights to the inlet enthalpy, so a
    species only in an inlet without flow needs no data for that enthalpy
    (``cp_liq``); constructing the inlet still needs the ``rho_liq`` of the
    species it holds.

    A flow with one entry (a 0-d or ``(1,)`` array) paired with one
    composition, shape ``(num_species,)``, is a scalar flow; only a flow
    with one entry per composition row is a profile. The evaporators do
    not support ``(1,)``-shaped feed flows yet: their initialization and
    energy balances fail on them (issue #520). One composition row of
    shape ``(1, num_species)`` paired with a flow profile (the ``Mixer``
    stores a time-independent composition so, next to an inlet that sets
    a time grid) applies at every row: a species in it is fed if any row
    flows, and the weights keep the shape of the composition, as the
    property kernels require.
    """
    fractions = np.asarray(fractions, dtype=float)  # [-]
    flowing = np.asarray(flow) != 0  # [-], per row
    if flowing.size == 1 and fractions.ndim == 1:
        flowing = flowing.reshape(())  # [-], a scalar flow
    elif (fractions.ndim == 2 and fractions.shape[0] == 1
          and flowing.size > 1):
        # One composition row for a flow profile: fed if any row flows.
        flowing = flowing.any()  # [-]
    elif flowing.ndim > 0:
        flowing = flowing[:, np.newaxis]  # [-], one flag per row

    return ((fractions != 0) & flowing).astype(float)


class StructuralPresenceMixin:
    """Pass a structural species-presence mask to the property kernels.

    A unit operation that carries every species as a solver state weights
    the property kernels it evaluates on those states (``weights=``) with
    ``_presence``, a ``SpeciesPresence`` record built from its charge and
    from the feed rows that flow, never from states: an absent species
    needs no property data (issue #425). Without a record (phases
    installed by hand, as some tests do) the kernels keep the
    fraction-based rule.

    Attributes
    ----------
    _presence_phase_names : tuple of str
        Names of the attributes holding the phases that a solve may
        overwrite, for example ``('Liquid_1', 'Vapor_1')``; a failed solve
        restores their complete state in place (``_solve_snapshot``). A
        class attribute that each unit sets.
    """

    _presence_phase_names: Tuple[str, ...] = ()

    def _start_presence(self, num_species: int,
                        *amounts: np.ndarray) -> None:
        """Start a new presence record from the given exact inputs.

        Parameters
        ----------
        num_species : int
            Number of species of the unit's states [-].
        *amounts : array-like
            Charge or feed fractions [-] or amounts, shape
            ``(num_species,)`` or ``(num_rows, num_species)``; a species
            with a nonzero entry is present from the start.
        """
        self._presence = SpeciesPresence(num_species)
        self._presence.record(*amounts)

    @staticmethod
    def _feed_weights(flow: Union[float, np.ndarray],
                      fractions: np.ndarray) -> np.ndarray:
        """Flag the feed entries that actually flow (see ``feed_weights``).

        Parameters
        ----------
        flow : float or array-like
            Effective feed flow [mol/s] or [kg/s], scalar or shape
            ``(num_rows,)``, after any shut-off.
        fractions : array-like
            Feed mole or mass fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``.

        Returns
        -------
        numpy.ndarray
            0/1 feed weights [-] of ``feed_weights``.
        """
        return feed_weights(flow, fractions)

    def _record_feed(self, flow: Union[float, np.ndarray],
                     fractions: np.ndarray,
                     time: Optional[Union[float, np.ndarray]] = None
                     ) -> np.ndarray:
        """Add the species of the flowing feed rows to the record.

        Parameters
        ----------
        flow : float or array-like
            Effective feed flow [mol/s] or [kg/s], scalar or shape
            ``(num_rows,)``, after any shut-off.
        fractions : array-like
            Feed mole or mass fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``.
        time : float or array-like, optional
            Absolute evaluation time [s] of the feed, scalar or one per row;
            it becomes the species' latch time if earlier than a recorded
            one. A single feed row evaluated for several times (controls
            that ignore the time) latches at the earliest of them. None
            records the species from the start.

        Returns
        -------
        numpy.ndarray
            The feed weights of ``_feed_weights`` [-]; a species becomes
            present only where they are nonzero, and stays present
            (latched). Without a record nothing is recorded.
        """
        fed = self._feed_weights(flow, fractions)  # [-]
        presence = getattr(self, '_presence', None)
        if presence is not None:
            row_time = time
            if np.ndim(time) > 0 and np.ndim(fed) == 1:
                # One feed row for a time profile (controls that ignore the
                # time): it applies at every time, so it latches at the first.
                row_time = float(np.min(time))  # [s]
            presence.record(fed, time=row_time)

        return fed

    @staticmethod
    def _phase_state(phase) -> dict:
        """Copy the complete state of a phase, stream or unit object.

        Parameters
        ----------
        phase : LiquidPhase, VaporPhase, a stream, or a unit operation
            Object whose attributes a solve may change.

        Returns
        -------
        dict
            The object's attributes, with arrays [units as stored], lists
            and dicts copied so that later in-place changes do not reach
            them; other values (scalars, strings, references to other
            objects such as controllers) are kept as they are.

        Notes
        -----
        Phases hold their property data, composition, amounts and derived
        fields (mass fractions, mean molar mass, flows) as plain
        attributes, so restoring them restores a consistent phase. A unit's
        results are replaced or appended to (``profiles_runs``), never
        changed inside, so one level of copying suffices for it too.
        """
        return {name: (value.copy()
                       if isinstance(value, (np.ndarray, list, dict))
                       else value)
                for name, value in vars(phase).items()}

    def _solve_snapshot(self) -> dict:
        """Record what a failed solve must leave unchanged.

        Returns
        -------
        dict
            The unit's own attributes (``'unit'``, as ``_phase_state``
            copies them: elapsed time, terminal states, profiles, results,
            duties, outlet, phase references), a copy of the presence record
            (``'presence'``, None without one), and the complete state of
            each phase named in ``_presence_phase_names`` (``'states'``),
            which the unit's model, initialization and result publication
            may overwrite.
        """
        presence = getattr(self, '_presence', None)
        phases = [getattr(self, name) for name in self._presence_phase_names]
        return {
            'unit': self._phase_state(self),
            'presence': None if presence is None else copy.deepcopy(presence),
            'states': [(phase, self._phase_state(phase)) for phase in phases]}

    def _restore_solve_snapshot(self, snapshot: dict) -> None:
        """Undo what a failed solve changed in the unit and its phases.

        Parameters
        ----------
        snapshot : dict
            Record returned by ``_solve_snapshot`` before the solve.

        Notes
        -----
        A failed solve keeps no results: whether the initialization, the
        integration or the publication of the results raised, the unit's
        attributes are restored in place, so a partly published segment
        (elapsed time, terminal states, profiles, results, duties, outlet)
        is withdrawn. The species that only its attempted integration
        latched (a feed evaluated past the failure, for example) must not
        stay present, while species latched by earlier, successful solves
        are in the snapshot and remain. The named phases are restored too,
        attribute by attribute and in place, so that every reference to
        them sees the restored state: a model that writes iterates into its
        phases would otherwise leave the next initialization starting from
        the last iterate, with inconsistent derived fields (fractions of
        one state, mass fractions of another).
        """
        vars(self).clear()
        vars(self).update(snapshot['unit'])
        self._presence = snapshot['presence']
        for phase, state in snapshot['states']:
            # In place, so that every reference to the phase sees it.
            vars(phase).clear()
            vars(phase).update(state)

    def _presence_weights(self, fractions: np.ndarray,
                          used: Optional[np.ndarray] = None,
                          time: Optional[Union[float, np.ndarray]] = None
                          ) -> Optional[np.ndarray]:
        """Return the kernel weights for some fractions.

        Parameters
        ----------
        fractions : array-like
            Mole or mass fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``.
        used : array-like of bool, optional
            Species whose results the caller uses, shape
            ``(num_species,)``.
        time : float or array-like, optional
            Absolute time [s] of the fractions, scalar or one per row: only
            species recorded at or before it count, so that a replay of
            earlier rows uses the presence of their time. None counts every
            recorded species.

        Returns
        -------
        numpy.ndarray or None
            0/1 presence weights [-] with the shape of ``fractions``. Without
            a presence record: ``None``, or ``fractions * used`` when
            ``used`` is given, the fraction-based rule.
        """
        presence = getattr(self, '_presence', None)
        if presence is None:
            return None if used is None else fractions * used

        return presence.weights(fractions, used, time=time)

    def _project_absent(self, fractions: np.ndarray,
                        time: Optional[Union[float, np.ndarray]] = None
                        ) -> np.ndarray:
        """Zero the round-off fractions of absent species.

        Parameters
        ----------
        fractions : array-like
            Mole or mass fractions [-] from solver states, shape
            ``(num_species,)`` or ``(num_rows, num_species)``.
        time : float or array-like, optional
            Absolute time [s] of the fractions, scalar or one per row (see
            ``_presence_weights``).

        Returns
        -------
        numpy.ndarray
            ``fractions`` with absent species at exactly zero, used to build
            the published phases, which would otherwise need data for those
            species; unchanged (as a float array) without a presence record.
        """
        presence = getattr(self, '_presence', None)
        if presence is None:
            return np.asarray(fractions, dtype=float)

        return presence.project(fractions, time=time)
