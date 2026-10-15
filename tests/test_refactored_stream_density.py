"""Densities of refactored phases and their dynamic hand-off to legacy units.

A refactored phase must give its density at the composition the caller asks
for, on either basis, and fall back to its own only when none is given. A
refactored continuous reactor's outlet must reach a legacy unit with its
volumetric flow converted at each sample's own density (#219). The reactor
solves with SciPy LSODA, so no test here needs Assimulo.
Expected densities are ideal mixing of the pure-liquid densities read
straight from the five-species ``tests/Flowsheet/data`` database.
"""

import json
import os

import numpy as np
import pytest

from PharmaPy.Connections import Connection
from PharmaPy.Containers import DynamicCollector
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import ContinuousReactor
from PharmaPy.Streams_Refactored import LiquidStream

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

TEMP = 313.15  # [K]
# A, B, C, D, solvent. The feed differs from the holdup, so the outlet
# composition, and with it the density, changes along the run.
HOLDUP_CONC = np.array([0.2, 0.2, 0.0, 0.0, 0.0])  # [mol/L]
FEED_CONC = np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # [mol/L]
HOLDUP_VOL = 0.01  # [m**3]
FEED_VOL_FLOW = 1e-5  # [m**3/s], a 1000 s residence time
RUNTIME = 600.0  # [s]
# A + B --> C fast enough to move the outlet composition within RUNTIME.
RATE_CONSTANT = np.array([1e-2])  # [L/(mol s)]
ACTIVATION_ENERGY = np.array([1e2])  # [J/mol]

# Mole fractions unlike the stored composition, one row per sample.
REQUESTED_MOLE_FRAC = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.8, 0.0, 0.0],
])  # [-]
# Both sides evaluate the same ideal-mixing sum in double precision.
DENSITY_RTOL = 1e-12  # [-]
# The outlet density must vary by more than this, or a single density for
# every sample would pass unnoticed. This run moves it by about 0.6 %.
DENSITY_SPREAD_FLOOR = 1e-3  # [-]


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


def test_refactored_outlet_reaches_a_legacy_unit_at_each_sample_density():
    """The collector's mass flow is each sample's volume flow times density."""
    reactor = ContinuousReactor(integrator=ScipyBackend(method="LSODA"),
                                isothermal=True)
    reactor.Phases = LiquidPhase(DATA_PATH, temp=TEMP,
                                 mole_conc=HOLDUP_CONC.copy(),
                                 vol=HOLDUP_VOL, name_solv="solvent")
    reactor.Inlet = LiquidStream(DATA_PATH, temp=TEMP,
                                 mole_conc=FEED_CONC.copy(),
                                 vol_flow=FEED_VOL_FLOW, name_solv="solvent")
    reactor.RxnKinetics = RxnKinetics(
        path=DATA_PATH, rxn_list=["A + B --> C"],
        k_params=RATE_CONSTANT, ea_params=ACTIVATION_ENERGY,
    )
    reactor.solve_unit(runtime=RUNTIME, verbose=False)

    connection = Connection(reactor, DynamicCollector())
    connection.transfer_data()

    upstream = connection.Matter.y_upstream
    mole_conc = np.asarray(upstream["mole_conc"])  # [mol/L]
    density = _ideal_mixing_density(
        mole_conc / mole_conc.sum(axis=1, keepdims=True))  # [kg/m**3]
    vol_flow = np.asarray(upstream["vol_flow"])  # [m**3/s]

    assert np.ptp(density) / density.mean() > DENSITY_SPREAD_FLOOR
    np.testing.assert_allclose(
        connection.Matter.y_inlet["mass_flow"], vol_flow * density,
        rtol=DENSITY_RTOL)

    # The conversion borrows the outlet's phase; the reactor's own keeps no
    # connection state.
    assert reactor.Outlet.Phases[0].y_upstream is None
