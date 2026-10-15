"""Densities of refactored phases at a requested composition.

A refactored phase must give its density at the composition the caller asks
for, on either basis, and fall back to its own only when none is given.
Expected densities are ideal mixing of the pure-liquid densities read
straight from the five-species ``tests/Flowsheet/data`` database.
"""

import json
import os

import numpy as np
import pytest

from PharmaPy.Streams_Refactored import LiquidStream

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

TEMP = 313.15  # [K]
# A, B, C, D, solvent.
FEED_CONC = np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # [mol/L]
FEED_VOL_FLOW = 1e-5  # [m**3/s]

# Mole fractions unlike the stored composition, one row per sample.
REQUESTED_MOLE_FRAC = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.8, 0.0, 0.0],
])  # [-]
# Both sides evaluate the same ideal-mixing sum in double precision.
DENSITY_RTOL = 1e-12  # [-]


def _ideal_mixing_density(mole_frac):
    """Return the ideal-mixing liquid density of each composition row.

    Parameters
    ----------
    mole_frac : numpy.ndarray
        Mole fractions [-], shape (num_rows, num_species).

    Returns
    -------
    numpy.ndarray
        Density [kg/m**3], shape (num_rows,), from
        ``1 / rho = sum_i w_i / rho_i`` with database ``mw`` [g/mol] and
        ``rho_liq`` [kg/m**3].
    """
    with open(DATA_PATH) as handle:
        database = json.load(handle)

    molar_mass = np.array([props["mw"] for props in database.values()])  # [g/mol]
    pure_density = np.array(
        [props["rho_liq"] for props in database.values()])  # [kg/m**3]

    mass = mole_frac * molar_mass  # [g per mol of mixture]
    mass_frac = mass / mass.sum(axis=1, keepdims=True)  # [-]

    return 1.0 / (mass_frac / pure_density).sum(axis=1)


def test_refactored_phase_density_follows_a_requested_mole_fraction():
    """A mole-fraction request is not overridden by the stored composition."""
    stream = LiquidStream(DATA_PATH, temp=TEMP, mole_conc=FEED_CONC.copy(),
                          vol_flow=FEED_VOL_FLOW, name_solv="solvent")

    requested = stream.getDensity(mole_frac=REQUESTED_MOLE_FRAC)  # [kg/m**3]

    np.testing.assert_allclose(
        requested, _ideal_mixing_density(REQUESTED_MOLE_FRAC),
        rtol=DENSITY_RTOL)

    # With no request, the stream's own composition, solvent filled in.
    stored_mole_frac = np.atleast_2d(stream.mole_frac)  # [-]
    np.testing.assert_allclose(
        stream.getDensity(), _ideal_mixing_density(stored_mole_frac)[0],
        rtol=DENSITY_RTOL)

