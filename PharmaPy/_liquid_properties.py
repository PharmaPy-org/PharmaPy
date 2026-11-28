"""Internal property inputs for liquid construction."""

from dataclasses import dataclass
from math import fsum, inf, isfinite, ulp

import numpy as np


@dataclass(frozen=True)
class BulkLiquidDensity:
    """Sourced bulk mass density for a single liquid reference state.

    Parameters
    ----------
    mass_density : float
        Unmasked, positive, finite mass per volume of the whole solution [kg/m**3].
    source : str
        Nonempty provenance for the density and its reference conditions,
        such as a measurement record or published datum.

    Raises
    ------
    ValueError
        If the density is invalid or the source is empty.

    Notes
    -----
    The density applies at the composition, temperature and pressure at
    which it was obtained. The caller supplies compatible solute
    concentrations and initial phase conditions. This value is used once
    to complete a single-solvent inventory and carries no model for state
    changes.
    """

    mass_density: float  # [kg/m**3], whole solution at its reference state
    source: str

    def __post_init__(self) -> None:
        """Validate the supplied scalar density and its provenance.

        Raises
        ------
        ValueError
            If the density is masked or is outside the finite positive
            scalar domain [kg/m**3], or the source is not a nonempty string.
        """
        if np.ma.is_masked(self.mass_density):
            raise ValueError("mass_density must not be masked.")
        value = np.asarray(self.mass_density)  # [kg/m**3]
        if value.ndim != 0 or value.dtype.kind not in 'iuf':
            raise ValueError(
                "mass_density must be a finite positive scalar [kg/m**3]."
            )
        density = float(value)  # [kg/m**3]
        if not isfinite(density) or density <= 0:
            raise ValueError(
                "mass_density must be a finite positive scalar [kg/m**3]."
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a nonempty string.")
        object.__setattr__(self, 'mass_density', density)

    def _solvent_mass_concentration(
            self, solute_mass_conc: np.ndarray, *, from_molar: bool) -> float:
        """Close the mass-per-volume balance at the reference state.

        Parameters
        ----------
        solute_mass_conc : numpy.ndarray
            Solute mass concentrations [kg/m**3], shape ``(num_solutes,)``,
            excluding the solvent. An empty array describes pure solvent.
        from_molar : bool
            Whether the solute mass concentrations were converted from
            molar concentrations [mol/L] using molar masses [g/mol].

        Returns
        -------
        float
            Solvent mass concentration [kg/m**3], equal to the bulk mass
            density less the sum of the solute mass concentrations.

        Raises
        ------
        ValueError
            If the solute concentrations are negative, nonfinite or have
            a sum greater than the bulk mass density beyond floating-point
            rounding at a zero residual.

        Notes
        -----
        Only a negative residual within the binary64 rounding allowance is
        clipped. Solute concentrations are retained unchanged.
        """
        if not np.all(np.isfinite(solute_mass_conc)) or np.any(solute_mass_conc < 0):
            raise ValueError("Solute concentrations must be finite and non-negative.")
        try:
            solute_total = fsum(solute_mass_conc)  # [kg/m**3]
        except OverflowError:
            solute_total = inf  # [kg/m**3], greater than any valid bulk density
        residual = self.mass_density - solute_total  # [kg/m**3]
        # Near exhaustion, allow one binary64 ULP at the density scale for
        # each rounding stage. For nonnegative terms the input errors add
        # in proportion to their contributions. On the mass basis the
        # stages are input representation, fsum and density representation.
        # The molar basis also represents each molar mass and rounds the
        # concentration-by-molar-mass product.
        rounding_stages = 5 if from_molar else 3  # [-]
        rounding = rounding_stages * ulp(self.mass_density)  # [kg/m**3]
        if residual < -rounding:
            raise ValueError(
                f"Solute concentrations sum to {solute_total!r} kg/m**3 and "
                f"exceed the bulk liquid mass density of {self.mass_density!r} kg/m**3."
            )
        return max(0.0, residual)
