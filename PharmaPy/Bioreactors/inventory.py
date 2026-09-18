"""Inventory feasibility for a fixed-step, input-declared metabolic source map."""

import numpy as np


class InventoryAvailability:
    """Constrain end-of-step amounts using the same population exposure as propagation."""

    def __init__(self, amounts, source_matrix, other_increment, *, viable,
                 step_day, cell_flux_scale, growth, death,
                 growth_index=None, growth_conversion=1.0):
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
        if (not np.isfinite([viable, step_day, cell_flux_scale, growth, death,
                             growth_conversion]).all()
                or viable < 0 or step_day <= 0 or cell_flux_scale <= 0
                or growth_conversion <= 0):
            raise ValueError('invalid fixed-step population exposure')
        if growth_index is not None and not 0 <= growth_index < self.matrix.shape[1]:
            raise ValueError('growth index is outside the declared flux vector')
        self.viable = float(viable)
        self.step_day = float(step_day)
        self.cell_flux_scale = float(cell_flux_scale)
        self.growth = float(growth)
        self.death = float(death)
        self.growth_index = growth_index
        self.growth_conversion = float(growth_conversion)
        # Row scaling is numerical only; it adds no nutrient or feasible slack.
        self.scale = np.maximum(np.abs(self.amounts), 1.0)

    def _exposure(self, fluxes):
        growth = self.growth if self.growth_index is None else (
            fluxes[self.growth_index] * self.growth_conversion)
        exponential = np.exp((growth - self.death) * self.step_day)
        factor = self.viable * self.step_day * self.cell_flux_scale / 2.
        return factor * (1. + exponential), factor * exponential

    def residual(self, fluxes):
        """Return scaled end inventories, which must all be nonnegative."""
        exposure, _ = self._exposure(fluxes)
        return (self.amounts + self.other + exposure * (self.matrix @ fluxes)) / self.scale

    def jacobian(self, fluxes):
        """Differentiate both mapped fluxes and growth-dependent population exposure."""
        exposure, exponential_part = self._exposure(fluxes)
        derivative = exposure * self.matrix.copy()
        if self.growth_index is not None:
            derivative[:, self.growth_index] += (
                self.matrix @ fluxes) * exponential_part * self.step_day * self.growth_conversion
        return derivative / self.scale[:, None]

    def linear_relaxation(self, lower, upper):
        """Bound uptake using minimum exposure, without excluding feasible solutions."""
        growth = self.growth if self.growth_index is None else (
            lower[self.growth_index] * self.growth_conversion)
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
