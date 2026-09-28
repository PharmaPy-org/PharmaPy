# -*- coding: utf-8 -*-
"""
Crystal size distribution conversions.

A size distribution can be written three ways, and PharmaPy uses all three at
different boundaries, which is where most of the confusion between the legacy
and the refactored stacks comes from:

**Quantity** - what the array counts.
    A *number density* is crystals per micron per unit of basis. A *volume
    fraction* or *mass fraction* is a shape: how the solid's volume or mass is
    spread over size, normalized to sum to one. Converting between them is not
    a rescale, it changes the shape by ``kv * x**3``, so no single multiplier
    undoes it.

**Extent** - what the basis is.
    An *extensive* distribution is an absolute count for a whole phase. An
    *intensive* one is per m3 of slurry. Converting between them IS a pure
    rescale, by the slurry volume.

Intensive number density is the standard everywhere inside the refactored
stack; extensive and fraction-based forms are input and output formats only,
converted at the boundary. The legacy ``SolidPhase`` stores the extensive
number density and accepts a volume or mass fraction on construction.

These helpers take explicit arrays rather than a phase object so the legacy
phase, the refactored mechanisms and anything comparing the two can share one
copy of the arithmetic and cannot drift apart.
"""

import numpy as np

eps = np.finfo(float).eps * 1.1

# Crystal volume is kv * x**3 with x in microns, so a volume in m3 picks up
# (1e-6)**3. Spelled out rather than left as a bare 1e18 in six places.
MICRON3_PER_M3 = 1e18


# ----------------------------------------------------------------------
# Extent: extensive <-> intensive
# ----------------------------------------------------------------------

def to_intensive(distrib, vol_slurry):
    """
    Absolute counts to a number density per m3 of slurry.

    This is the conversion into the refactored stack's storage convention.
    """

    vol_slurry = float(vol_slurry)

    if vol_slurry <= 0:
        raise ValueError(
            "Converting between extensive and intensive needs a slurry "
            "volume, and it must be positive. A solid on its own does not "
            "have one - the slurry volume is the liquid's, divided by "
            "(1 - solid volume fraction) - so pass the value explicitly when "
            "the mechanism is not attached to a phase inside a vessel."
        )

    return np.asarray(distrib, dtype=float) / vol_slurry


def to_extensive(distrib, vol_slurry):
    """Number density per m3 of slurry back to absolute counts."""

    vol_slurry = float(vol_slurry)

    if vol_slurry <= 0:
        raise ValueError(
            "Converting between extensive and intensive needs a slurry "
            "volume, and it must be positive. A solid on its own does not "
            "have one - the slurry volume is the liquid's, divided by "
            "(1 - solid volume fraction) - so pass the value explicitly when "
            "the mechanism is not attached to a phase inside a vessel."
        )

    return np.asarray(distrib, dtype=float) * vol_slurry


# ----------------------------------------------------------------------
# Quantity: fraction shapes <-> number density
# ----------------------------------------------------------------------

def volume_fraction_to_number(x_grid, dx, vol_fraction, mass, density, kv=1):
    """
    A volume-fraction shape to a number density, in #/micron.

    The solid occupies ``mass/density`` of volume; ``vol_fraction`` says how
    that volume is spread over size; dividing by the volume of one crystal in
    each bin, ``kv * x**3``, turns volume into a count.

    This is `SolidPhase.convert_distribution`'s `vol_distr` branch, term for
    term, and the legacy method now calls through to here.
    """

    x_grid = np.asarray(x_grid, dtype=float)

    if mass == 0:
        raise ValueError(
            "A volume fraction is only a shape, so a mass is needed to give "
            "it a magnitude."
        )

    return ((mass / density) * np.asarray(vol_fraction, dtype=float)
            / kv / x_grid**3 / dx * MICRON3_PER_M3)


def number_to_volume_fraction(x_grid, dx, number_density, mom_three, kv=1):
    """
    A number density to a volume-fraction shape.

    The inverse of `volume_fraction_to_number`, normalized by the third
    moment so the result sums to one. `SolidPhase.convert_distribution`'s
    `num_distr` branch.
    """

    x_grid = np.asarray(x_grid, dtype=float)

    mom_three = np.asarray(mom_three, dtype=float).copy()
    mom_three[mom_three == 0] = eps

    return (np.asarray(number_density, dtype=float)
            * dx * x_grid**3 * kv / mom_three / MICRON3_PER_M3)


def mass_fraction_to_number(x_grid, mass_fraction, mass, kv=1):
    """
    A mass-fraction shape to a number density.

    `SolidPhase.getDistribution`'s `mass_perc` branch, reproduced as it
    stands. Note it carries neither `dx` nor the density, unlike the volume
    branch above - that asymmetry is the original's, kept here rather than
    quietly corrected, because changing it would move any result that uses
    `distrib_type='mass_perc'`.
    """

    x_grid = np.asarray(x_grid, dtype=float)

    return (mass * np.asarray(mass_fraction, dtype=float)
            / x_grid**3 / kv * MICRON3_PER_M3)
