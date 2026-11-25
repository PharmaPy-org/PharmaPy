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
        Positive finite mass per volume of the whole solution [kg/m**3].
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
            If the density is not a finite positive scalar [kg/m**3] or
            the source is not a nonempty string.
        """
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

    def _solvent_mass_concentration(self, solute_mass_conc: np.ndarray) -> float:
        """Close the mass-per-volume balance at the reference state.

        Parameters
        ----------
        solute_mass_conc : numpy.ndarray
            Solute mass concentrations [kg/m**3], shape ``(num_solutes,)``,
            excluding the solvent. An empty array describes pure solvent.

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
        """
        if not np.all(np.isfinite(solute_mass_conc)) or np.any(solute_mass_conc < 0):
            raise ValueError("Solute concentrations must be finite and non-negative.")
        try:
            solute_total = fsum(solute_mass_conc)  # [kg/m**3]
        except OverflowError:
            solute_total = inf  # [kg/m**3], greater than any valid bulk density
        residual = self.mass_density - solute_total  # [kg/m**3]
        # Accurate summation still rounds differently from a supplied decimal
        # density at solvent exhaustion. Allow one representable step there.
        rounding = ulp(self.mass_density)  # [kg/m**3]
        if residual < -rounding:
            raise ValueError(
                f"Solute concentrations sum to {solute_total:g} kg/m**3 and "
                f"exceed the bulk liquid mass density of {self.mass_density:g} kg/m**3."
            )
        return max(0.0, residual)
