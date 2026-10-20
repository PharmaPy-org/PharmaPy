"""Inventory feasibility for accepted metabolic source maps."""

import numpy as np


class LinearInventoryAvailability:
    """Require nonnegative inventories or boundary rates under an affine source map.

    The constraint is ``amounts + other + matrix @ fluxes >= 0`` row by row. The
    caller chooses a consistent amount basis (``PathwayMetabolism`` uses [mol],
    ``RateReconciledCulture`` uses [mmol]). Over a fixed step, rows are end
    inventories; at a depletion boundary, rows are net rates of exhausted pools.

    Parameters
    ----------
    amounts : array_like, shape (n_species,)
        Current inventories [amount], or zeros for boundary rate rows.
    source_matrix : array_like, shape (n_species, n_fluxes)
        Inventory change per unit flux [amount per flux unit] over the step,
        or [amount/time per flux unit] at a boundary.
    other_increment : array_like, shape (n_species,)
        Flux-independent change: supply, transfer and degradation [amount] over
        the step, or [amount/time] at a boundary.
    growth_index : int, optional
        Position of the growth flux in the flux vector, if growth is a flux.

    Raises
    ------
    ValueError
        Inputs are nonfinite, empty, misaligned, or ``growth_index`` is outside
        the flux vector.
    """

    def __init__(self, amounts, source_matrix, other_increment, *, growth_index=None):
        """Store the affine availability map and its row scales.

        Parameters
        ----------
        amounts : array_like, shape (n_species,)
            Current inventories [amount] or zero boundary inventories.
        source_matrix : array_like, shape (n_species, n_fluxes)
            Inventory change per unit flux [amount per flux unit].
        other_increment : array_like, shape (n_species,)
            Flux-independent inventory change [amount].
        growth_index : int, optional
            Position of the growth flux, if declared.

        Raises
        ------
        ValueError
            Inputs are nonfinite, empty or misaligned, or ``growth_index`` is out
            of range.
        """
        self.amounts = np.asarray(amounts, dtype=float).copy()
        self.matrix = np.asarray(source_matrix, dtype=float).copy()
        self.other = np.asarray(other_increment, dtype=float).copy()
        if (self.amounts.ndim != 1 or self.matrix.ndim != 2
                or not self.amounts.size or not self.matrix.shape[1]
                or self.matrix.shape[0] != self.amounts.size
                or self.other.shape != self.amounts.shape
                or not all(np.isfinite(x).all() for x in
                           (self.amounts, self.matrix, self.other))):
            raise ValueError('inventory amounts and source mappings must be finite and aligned')
        if growth_index is not None and not 0 <= growth_index < self.matrix.shape[1]:
            raise ValueError('growth index is outside the declared flux vector')
        self.growth_index = growth_index
        # Row scale [amount]: the largest term in each row. The 1e-30 floor only
        # prevents division by zero for an all-zero row; it is far below any
        # physical inventory, so it never changes a nonzero row's scale.
        self.scale = np.maximum.reduce((np.abs(self.amounts), np.abs(self.other),
                                       np.max(np.abs(self.matrix), axis=1),
                                       np.full(self.amounts.shape, 1e-30)))

    @classmethod
    def boundary(cls, amounts, source_matrix, other_rates, *, growth_index=None):
        """At exhausted inventories require net production, including actual supply.

        Parameters
        ----------
        amounts : array_like, shape (n_species,)
            Current inventories [amount]; rows with ``amount <= 0`` are exhausted.
        source_matrix : array_like, shape (n_species, n_fluxes)
            Instantaneous source rate per unit flux [amount/time per flux unit].
        other_rates : array_like, shape (n_species,)
            Flux-independent rates, such as feeds and transfer [amount/time].
        growth_index : int, optional
            Position of the growth flux, if declared.

        Returns
        -------
        LinearInventoryAvailability or None
            Rate constraint for exhausted rows only, or None when no pool is empty.
        """
        rows = np.asarray(amounts) <= 0.
        if not rows.any():
            return None
        return cls(np.zeros(rows.sum()), np.asarray(source_matrix)[rows],
                   np.asarray(other_rates)[rows], growth_index=growth_index)

    def residual(self, fluxes):
        """Return scaled constraint values, which must all be nonnegative.

        Parameters
        ----------
        fluxes : ndarray, shape (n_fluxes,)
            Candidate fluxes in the source-matrix flux basis.

        Returns
        -------
        ndarray, shape (n_rows,)
            ``(amounts + other + matrix @ fluxes) / scale`` [-].
        """
        return (self.amounts + self.other + self.matrix @ fluxes) / self.scale

    def _exposure(self, fluxes):
        """Return the population exposure multiplying the source map.

        Parameters
        ----------
        fluxes : ndarray, shape (n_fluxes,)
            Candidate fluxes; unused because population and time are already in
            the linear map.

        Returns
        -------
        tuple of float
            ``(exposure, d_exposure_part)`` = ``(1, 0)`` [-].
        """
        # Population and time factors are already included in the linear map.
        return 1., 0.

    def jacobian(self, fluxes):
        """Return the constant derivative of the scaled residual.

        Parameters
        ----------
        fluxes : ndarray, shape (n_fluxes,)
            Candidate fluxes; unused because the map is affine.

        Returns
        -------
        ndarray, shape (n_rows, n_fluxes)
            ``matrix / scale`` [1/flux unit].
        """
        return self.matrix / self.scale[:, None]

    def linear_relaxation(self, lower, upper):
        """Return the exact linear inequality ``A @ fluxes <= b`` for this map.

        Parameters
        ----------
        lower, upper : ndarray or None
            Flux bounds; unused because the affine constraint is already linear.

        Returns
        -------
        tuple of ndarray
            ``(A, b)`` with ``A = -matrix / scale`` [1/flux unit] and
            ``b = (amounts + other) / scale`` [-].
        """
        return -self.matrix / self.scale[:, None], (self.amounts + self.other) / self.scale


class InventoryAvailability(LinearInventoryAvailability):
    """Constrain end amounts using the same population exposure as propagation.

    Over one fixed step the biological source is ``exposure * matrix @ fluxes``,
    where exposure is the trapezoidal time integral of the exponentially
    propagated population, ``viable * dt * scale * (1 + exp((mu - kd) dt)) / 2``.
    This matches ``RateReconciledCulture``'s population update, so accepted
    fluxes cannot overdraw an inventory before the step completes.

    Parameters
    ----------
    amounts : array_like, shape (n_species,)
        Current inventories [mmol].
    source_matrix : array_like, shape (n_species, n_fluxes)
        Species change per unit network flux per unit population exposure
        [-], mapping network fluxes to exchanges.
    other_increment : array_like, shape (n_species,)
        Supply and degradation over the step [mmol].
    viable : float
        Current population [10^6 cell] or biomass [kgDW], matching the flux
        normalization.
    step_day : float
        Step length [day].
    cell_flux_scale : float
        Conversion from network flux to [mmol/(population day)].
    growth, death : float
        Specific growth and death rates [1/day]; growth is replaced by the
        growth flux when ``growth_index`` is declared.
    growth_index : int, optional
        Position of the growth flux in the flux vector.
    growth_conversion : float, optional
        Growth flux to specific growth conversion [1/day per flux unit].
        Default 1.0.

    Raises
    ------
    ValueError
        Exposure inputs are nonfinite, the population is negative, or the step,
        flux scale or growth conversion is not positive.
    """

    def __init__(self, amounts, source_matrix, other_increment, *, viable,
                 step_day, cell_flux_scale, growth, death,
                 growth_index=None, growth_conversion=1.0):
        """Store the population exposure model and the affine inventory map.

        Parameters
        ----------
        amounts : array_like, shape (n_species,)
            Current inventories [mmol].
        source_matrix : array_like, shape (n_species, n_fluxes)
            Species change per unit network flux and exposure [-].
        other_increment : array_like, shape (n_species,)
            Flux-independent change over the step [mmol].
        viable : float
            Current population [10^6 cell] or [kgDW].
        step_day : float
            Step length [day].
        cell_flux_scale : float
            Network flux conversion [mmol/(population day) per flux unit].
        growth, death : float
            Specific growth and death rates [1/day].
        growth_index : int, optional
            Position of the growth flux.
        growth_conversion : float, optional
            Growth flux conversion [1/day per flux unit]. Default 1.0.

        Raises
        ------
        ValueError
            Exposure inputs are invalid.
        """
        super().__init__(amounts, source_matrix, other_increment, growth_index=growth_index)
        # Row scale [mmol]: relative for large pools, absolute (1 mmol) for
        # small ones so near-empty pools are judged on an absolute basis.
        self.scale = np.maximum(np.abs(self.amounts), 1.0)
        if (not np.isfinite([viable, step_day, cell_flux_scale, growth, death,
                             growth_conversion]).all()
                or viable < 0 or step_day <= 0 or cell_flux_scale <= 0
                or growth_conversion <= 0):
            raise ValueError('invalid fixed-step population exposure')
        self.viable = float(viable)
        self.step_day = float(step_day)
        self.cell_flux_scale = float(cell_flux_scale)
        self.growth = float(growth)
        self.death = float(death)
        self.growth_conversion = float(growth_conversion)

    def _exposure(self, fluxes):
        """Return trapezoidal population exposure and its exponential part.

        Parameters
        ----------
        fluxes : ndarray, shape (n_fluxes,)
            Candidate fluxes; the growth flux sets specific growth when declared.

        Returns
        -------
        tuple of float
            ``(exposure, exponential_part)`` in [population day x flux scale];
            the second term is the part whose derivative carries ``exp``.
        """
        growth = self.growth if self.growth_index is None else (
            fluxes[self.growth_index] * self.growth_conversion)
        exponential = np.exp((growth - self.death) * self.step_day)  # [-]
        # Trapezoid rule: mean of start and end populations times the step.
        factor = self.viable * self.step_day * self.cell_flux_scale / 2.
        return factor * (1. + exponential), factor * exponential

    def residual(self, fluxes):
        """Return scaled end inventories, which must all be nonnegative.

        Parameters
        ----------
        fluxes : ndarray, shape (n_fluxes,)
            Candidate network fluxes.

        Returns
        -------
        ndarray, shape (n_species,)
            End inventories divided by ``max(|amounts|, 1 mmol)`` [-].
        """
        exposure, _ = self._exposure(fluxes)
        return (self.amounts + self.other + exposure * (self.matrix @ fluxes)) / self.scale

    def jacobian(self, fluxes):
        """Differentiate both mapped fluxes and growth-dependent population exposure.

        Parameters
        ----------
        fluxes : ndarray, shape (n_fluxes,)
            Candidate network fluxes.

        Returns
        -------
        ndarray, shape (n_species, n_fluxes)
            Derivative of the scaled residual [1/flux unit].
        """
        exposure, exponential_part = self._exposure(fluxes)
        derivative = exposure * self.matrix.copy()
        if self.growth_index is not None:
            derivative[:, self.growth_index] += (
                self.matrix @ fluxes) * exponential_part * self.step_day * self.growth_conversion
        return derivative / self.scale[:, None]

    def linear_relaxation(self, lower, upper):
        """Bound uptake using minimum exposure, without excluding feasible solutions.

        Parameters
        ----------
        lower : ndarray, shape (n_fluxes,)
            Lower flux bounds; the growth entry sets minimum exposure.
        upper : ndarray, shape (n_fluxes,)
            Upper flux bounds; unused.

        Returns
        -------
        tuple
            ``(A, b)`` for ``A @ fluxes <= b`` [mmol/exposure], or ``(None, None)``
            when no row admits a valid relaxation.

        Notes
        -----
        For nonnegative remaining inventory, the minimum exposure gives a
        necessary, possibly loose, uptake bound. Rows with negative remaining
        inventory need the full nonlinear constraint and are not relaxed.
        """
        if self.growth_index is None:
            exposure, _ = self._exposure(np.zeros(len(lower)))
            return (-exposure * self.matrix / self.scale[:, None],
                    (self.amounts + self.other) / self.scale)
        growth = lower[self.growth_index] * self.growth_conversion
        exposure = (self.viable * self.step_day * self.cell_flux_scale / 2.
                    * (1. + np.exp((growth - self.death) * self.step_day)))
        remaining = self.amounts + self.other
        # For nonnegative remaining inventory, minimum exposure gives a necessary
        # (possibly loose) uptake bound. Negative remaining amounts still need
        # the full nonlinear production constraint; never reject them here.
        rows = remaining >= 0.
        if exposure == 0. or not rows.any():
            return None, None
        return -self.matrix[rows], remaining[rows] / exposure
