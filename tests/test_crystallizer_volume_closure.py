"""Volume closure of the per-volume population balance in a closed batch.

The refactored crystallizers carry their crystals per m**3 of slurry, so the
slurry volume has to be differentiated consistently. Crystallization moves
solute out of the liquid at its liquid density and into crystals at the
crystal density, which changes the slurry volume and so dilutes or
concentrates the population. Without that term the solid mass rebuilt from
the population drifted from the liquid's loss whenever the two densities
differed (PR #373 review); an equal-density fixture never showed it.

Fixture: 1 kg of liquid, 20 % C in solvent, seeded with 50 um crystals at a
crystal volume fraction of 0.1; constant solubility 50 kg/m**3; linear
growth, no nucleation; isothermal at 298.15 K. The crystal density is 1500
kg/m**3 against the database's liquid C at 1200 kg/m**3, with the latter as
the equal-density control.

Units: mass [kg], density [kg/m**3], moments [um**k/m**3 of slurry], number
density [#/(um m**3 of slurry)], volume [m**3], rates per second.

The finite-volume form carries a separate, pre-existing discrepancy between
its discrete growth flux and the analytic growth integral the mass transfer
uses, of order 1e-6 kg/s here and independent of the densities, so the exact
conservation identity is asserted on the moment form; the dilution term
itself is asserted on both forms.
"""

import os

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.DataClasses import PhaseRef
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import (MomentsPopulationBalance, OneDFVMMechanism,
                                 PopulationBalanceMechanism)
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

TARGET = "C"
TARGET_INDEX = 2  # position of C in the database's A, B, C, D, solvent
CHARGE_MASS = 1.0  # [kg]
CHARGE_MASS_FRAC = [0.0, 0.0, 0.2, 0.0, 0.8]  # [-]
CRYSTAL_MASS_FRAC = [0.0, 0.0, 1.0, 0.0, 0.0]  # [-], pure C
TEMP = 298.15  # [K]

SOLUBILITY = 50.0  # [kg/m**3], constant; the charge holds about 187 kg/m**3
GROWTH = (0.1, 0.0, 1.0)  # k_g [um/s], E_g [J/mol], g [-]: linear growth

SEED_SIZE = 50.0  # [um]
SEED_NUMBER = 8e11  # [#/m**3 of slurry]; kv * mu_3 * 1e-18 = 0.1 [-]
SHAPE_FACTOR = 1.0  # [-]

# The database's liquid density of C. Equal to the crystal density, the
# volume lost by the liquid equals the volume gained by the crystals and the
# slurry volume does not change, which is why that case never failed.
LIQUID_SOLUTE_DENSITY = 1200.0  # [kg/m**3]
# A distinct crystal density, chosen so the slurry volume shrinks.
CRYSTAL_DENSITY = 1500.0  # [kg/m**3]

FVM_GRID = np.arange(1.0, 200.0)  # [um], 1 um spacing
FVM_SEED_WIDTH = 5.0  # [um], standard deviation of the Gaussian seed

RUNTIME = 10.0  # [s], the review's reproduction; it lost 8.5e-4 kg
SOLVER_TOL = 1e-10  # [-], rtol and atol of the SciPy LSODA solve

# The initial-state identity is algebra on one right-hand side, so only
# roundoff separates the two sides; the missing term gave 2.8e-2.
ALGEBRAIC_RTOL = 1e-10  # [-]
# Over the solve each state carries the solver's 1e-10 relative error; a
# decade of margin. The missing term gave 7.2e-4.
SOLVE_CONSERVATION_RTOL = 1e-9  # [-]
# Anti-vacuity floor: this fixture crystallizes at about 3e-3 kg/s.
ACTIVE_MASS_RATE_FLOOR = 1e-4  # [kg/s]


def _seed_distribution():
    """Gaussian seed carrying ``SEED_NUMBER`` crystals on ``FVM_GRID``.

    Returns
    -------
    numpy.ndarray
        Number density [#/(um m**3 of slurry)], shape ``FVM_GRID.shape``.
    """
    return (SEED_NUMBER / (FVM_SEED_WIDTH * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((FVM_GRID - SEED_SIZE) / FVM_SEED_WIDTH) ** 2))


def _build(form, crystal_density, integrator=None):
    """Build a closed, isothermal batch crystallizer.

    Parameters
    ----------
    form : {'moments', 'fvm'}
        Whether the crystals are carried as moments or on a size grid.
    crystal_density : float
        Crystal density [kg/m**3].
    integrator : IntegratorBackend, optional
        Backend for a solve. None for solver-free evaluation.

    Returns
    -------
    tuple
        The compiled ``BatchCrystallizer``, its population balance and its
        liquid phase.
    """
    liquid = LiquidPhase(DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC,
                         temp=TEMP)
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    if form == "moments":
        mechanism = MomentsPopulationBalance(
            owning_phase=solid, target_components=TARGET, solvent_name="solvent",
            moments_init=SEED_NUMBER * SEED_SIZE ** np.arange(4),
            density=crystal_density, kv=SHAPE_FACTOR)
    else:
        mechanism = OneDFVMMechanism(
            solid, target_components=TARGET, solvent_name="solvent",
            x_grid=FVM_GRID, distrib_init=_seed_distribution(),
            density=crystal_density, kv=SHAPE_FACTOR)
    solid.mechanisms = mechanism

    vessel = BatchCrystallizer(integrator=integrator, isothermal=True)
    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = CrystKinetics(coeff_solub=[SOLUBILITY], growth=GROWTH)
    vessel.compile_structure()

    # Guard the fixture's premise against the shared database.
    liquid_densities, _ = liquid.getDensityPure("liquid")  # [kg/m**3]
    assert liquid_densities[TARGET_INDEX] == LIQUID_SOLUTE_DENSITY

    return vessel, mechanism, liquid


def _state_and_rates(vessel):
    """Evaluate the material balance once at t = 0 and split it by state.

    Parameters
    ----------
    vessel : BatchCrystallizer
        Compiled vessel.

    Returns
    -------
    tuple of numpy.ndarray
        Liquid species masses [kg], their rates [kg/s], the population and
        its rate, both per m**3 of slurry.
    """
    states = np.asarray(vessel.create_solver_init_states(), dtype=float)
    rates = np.asarray(vessel.unit_model(0.0, states, mat_bce=True), dtype=float).copy()

    collection = vessel.solver_state_collection
    species_key = next(k for k in collection.states if k.name == "mass_j")
    population_key = next(k for k in collection.states
                          if k.name in ("mu_n", "distrib"))

    return (states[collection.slices[species_key]],
            rates[collection.slices[species_key]],
            states[collection.slices[population_key]],
            rates[collection.slices[population_key]])


def _solid_mass_rate(mechanism, liquid, species_rates, population, population_rate):
    """Differentiate the solid mass implied by the per-volume population.

    Parameters
    ----------
    mechanism : PopulationBalanceMechanism
        Population balance, for the crystal density and volume fraction.
    liquid : LiquidPhase
        Liquid at the initial state.
    species_rates : numpy.ndarray
        Liquid species mass rates [kg/s].
    population, population_rate : numpy.ndarray
        Population per m**3 of slurry and its rate.

    Returns
    -------
    float
        d(solid mass)/dt [kg/s].

    Notes
    -----
    With phi the crystal volume fraction [-] and V_L the liquid volume
    [m**3], the solid mass is ``rho_s * phi * V_L / (1 - phi)``, so by the
    chain rule its rate is ``rho_s * (dphi/dt * V_L / (1 - phi)**2 +
    phi / (1 - phi) * dV_L/dt)``. The liquid volume is ``sum(m_i / rho_i)``
    under the database's ideal-volume mixing, and the volume fraction is
    linear in the population, so ``solid_volume_fraction`` of the rate is
    the fraction's rate. This uses only the definition of the per-volume
    basis, not the balance under test.
    """
    liquid_densities, _ = liquid.getDensityPure("liquid")  # [kg/m**3]
    vol_liquid = float(liquid.vol)  # [m**3]
    vol_liquid_rate = float(np.sum(species_rates / liquid_densities))  # [m**3/s]

    phi = mechanism.solid_volume_fraction(population)  # [-]
    phi_rate = mechanism.solid_volume_fraction(population_rate)  # [1/s]

    return mechanism.density * (phi_rate * vol_liquid / (1.0 - phi) ** 2
                                + phi / (1.0 - phi) * vol_liquid_rate)


@pytest.mark.parametrize(
    "crystal_density", [CRYSTAL_DENSITY, LIQUID_SOLUTE_DENSITY],
    ids=["unequal_densities", "equal_densities"],
)
def test_closed_batch_conserves_mass_at_the_initial_state(crystal_density):
    """The solid gains exactly the mass the liquid loses, from the first instant.

    Before the slurry-volume change of crystallization entered the
    population balance, the total mass rate was
    ``phi / (1 - phi) * (1 - rho_crystal / rho_solute)`` of the transfer
    rate, -8.1e-5 kg/s on this fixture, and zero only at equal densities.
    """
    vessel, mechanism, liquid = _build("moments", crystal_density)
    _, species_rates, population, population_rate = _state_and_rates(vessel)

    liquid_mass_rate = float(species_rates.sum())  # [kg/s]
    solid_mass_rate = _solid_mass_rate(
        mechanism, liquid, species_rates, population, population_rate)  # [kg/s]

    # Crystallization is active, and only the target leaves the liquid.
    assert -species_rates[TARGET_INDEX] > ACTIVE_MASS_RATE_FLOOR
    assert liquid_mass_rate == pytest.approx(species_rates[TARGET_INDEX])

    assert liquid_mass_rate + solid_mass_rate == pytest.approx(
        0.0, abs=ALGEBRAIC_RTOL * abs(liquid_mass_rate))


@pytest.mark.parametrize("form", ["moments", "fvm"])
def test_phase_transfer_dilutes_the_population(form):
    """The population is concentrated as the slurry volume shrinks.

    The kinetic terms of the population balance depend only on the growth
    rate and the population, not on the crystal density, and at equal
    densities the transfer leaves the slurry volume unchanged. So the
    difference between the two builds is the dilution term alone:
    ``-n * (dV/dt) / V`` with ``dV/dt = m_dot * (1/rho_crystal -
    1/rho_solute)``, the crystal volume gained less the liquid volume lost.
    """
    vessel, mechanism, liquid = _build(form, CRYSTAL_DENSITY)
    _, species_rates, population, population_rate = _state_and_rates(vessel)
    control, _, _ = _build(form, LIQUID_SOLUTE_DENSITY)
    _, _, control_population, control_rate = _state_and_rates(control)

    np.testing.assert_array_equal(control_population, population)

    transfer_rate = -float(species_rates[TARGET_INDEX])  # [kg/s] into crystals
    assert transfer_rate > ACTIVE_MASS_RATE_FLOOR

    vol_rate = transfer_rate * (1.0 / CRYSTAL_DENSITY
                                - 1.0 / LIQUID_SOLUTE_DENSITY)  # [m**3/s]
    phi = mechanism.solid_volume_fraction(population)  # [-]
    vol_slurry = float(liquid.vol) / (1.0 - phi)  # [m**3]
    expected_dilution = -(vol_rate / vol_slurry) * population

    # The slurry shrinks, so the term is nonzero and concentrates.
    assert vol_rate < 0
    np.testing.assert_allclose(
        population_rate - control_rate, expected_dilution, rtol=ALGEBRAIC_RTOL)


@pytest.mark.integration
def test_closed_batch_conserves_mass_over_a_solve():
    """Total mass is constant over a 10 s solve with unequal densities.

    The total is rebuilt from the published trajectories, the liquid species
    masses and the moments with the published slurry volume, rather than
    read from the vessel's own total, which is then checked against it.
    """
    vessel, mechanism, _ = _build(
        "moments", CRYSTAL_DENSITY,
        integrator=ScipyBackend(method="LSODA",
                                options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}),
    )
    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    species_mass = np.asarray(vessel.result.mass_j_liquid0)  # [kg]
    moments = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    vol_slurry = np.asarray(vessel.result.vessel_vol).ravel()  # [m**3]

    solid_mass = (mechanism.density * SHAPE_FACTOR * moments[:, 3]
                  * mechanism.VOLUME_UNIT_FACTOR * vol_slurry)  # [kg]
    total_mass = species_mass.sum(axis=1) + solid_mass  # [kg]

    # Guard against a vacuous pass: the batch did crystallize.
    assert solid_mass[-1] - solid_mass[0] > ACTIVE_MASS_RATE_FLOOR * RUNTIME
    np.testing.assert_allclose(total_mass, total_mass[0],
                               rtol=SOLVE_CONSERVATION_RTOL)
    np.testing.assert_allclose(
        np.asarray(vessel.result.Total_m_in_vessel).ravel(), total_mass,
        rtol=SOLVE_CONSERVATION_RTOL)
