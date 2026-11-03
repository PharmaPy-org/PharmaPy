"""QMOM population balance in a closed, isothermal batch crystallizer.

Nucleation, growth and dissolution run at constant rates (``CrystKinetics``
custom mechanisms reading their first parameter), so the references are
closed form: the moment equations on one right-hand side, and over a solve
the method of characteristics for seeds and for nuclei born at zero size.
Growth carries the size factor (1 + beta*L)**alpha. The crystal density
equals the liquid density of the solute, so the slurry volume stays
constant, except in the mass-balance test, which makes them differ.

Units: mass [kg], density [kg/m**3], sizes [um], moments [um**k/m**3 of
slurry], rates per second. The solves use SciPy LSODA, about 2 s in all.
"""

import os

import numpy as np
import pytest
from scipy.integrate import quad

from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import MomentsPopulationBalance, QMOMPopulationBalance
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
# Constant; the charge holds about 187 kg/m**3, so the batch stays
# supersaturated over the solves, and undersaturated at the higher value.
SOLUBILITY = 50.0  # [kg/m**3]
UNDERSATURATED_SOLUBILITY = 300.0  # [kg/m**3]
SHAPE_FACTOR = 1.0  # [-]
# The database's liquid density of C, and a distinct crystal density.
LIQUID_SOLUTE_DENSITY = 1200.0  # [kg/m**3]
CRYSTAL_DENSITY = 1500.0  # [kg/m**3]

GROWTH_RATE = 0.1  # G0 [um/s], at zero size
DISSOLUTION_RATE = -0.05  # D [um/s], negative as CrystKinetics returns it
NUCLEATION_RATE = 1e8  # B [#/(m**3 s)], at zero size
SIZE_FACTOR_SLOPE = 0.02  # beta [1/um]
SQUARE_ROOT_EXPONENT = 0.5  # alpha [-]
LINEAR_EXPONENT = 1.0  # alpha [-]

SEED_SIZE = 50.0  # [um]
SEED_COUNT = 1e9  # [#/m**3 of slurry]; kv * mu_3 * 1e-18 = 1.25e-4 [-]
NUM_MOM = 6  # [-], three nodes
RUNTIME = 600.0  # [s]; seeds reach 153 um and the first nuclei 78 um

# Assimulo CVode's default relative tolerance, which PharmaPy's integrator
# backends keep: the size of the error an integrated moment can carry.
CVODE_DEFAULT_RTOL = 1e-6  # [-]
# One right-hand side, compared through the inversion's rounding.
ALGEBRAIC_RTOL = 1e-10  # [-]
# rtol and atol of the LSODA solves.
SOLVER_TOL = 1e-10  # [-]
# Linear growth is exact under QMOM, so only integration error remains:
# at most 1.4e-10 over these solves (3 Nov 2026), a margin of 70.
EXACT_CLOSURE_RTOL = 1e-8  # [-]
# The three-node closure of the square-root law misses m_5 of the seeded
# batch by 2.5e-4 and of the unseeded one by 6.5e-5 (3 Nov 2026); a margin
# of four. A two-node closure misses the seeded m_3 by 1e-2, and dropping
# the nuclei node, as the inversion once did, the unseeded m_5 by 0.76.
SQUARE_ROOT_CLOSURE_RTOL = 1e-3  # [-]
# Anti-vacuity floor: these fixtures crystallize at 1e-6 to 1e-5 kg/s.
ACTIVE_MASS_RATE_FLOOR = 1e-8  # [kg/s]


def constant_rate(sup_sat, conc_sat, moments, temp, temp_ref, params):
    """Rate law independent of the driving force, for closed-form references.

    Parameters
    ----------
    sup_sat : float
        Supersaturation [-], ignored.
    conc_sat : float
        Solubility [kg/m**3], ignored.
    moments : numpy.ndarray
        Moments [m**k/m**3], ignored.
    temp : float
        Temperature [K], ignored.
    temp_ref : float
        Reference temperature [K], ignored.
    params : array-like
        The mechanism's parameters; the first is the rate.

    Returns
    -------
    float
        ``params[0]``: G0 [um/s] for growth, B [#/(m**3 s)] for nucleation.
    """
    return params[0]


def _kinetics(alpha, solubility=SOLUBILITY):
    """Constant-rate nucleation, growth and dissolution.

    Parameters
    ----------
    alpha : float or None
        Exponent of the size factor (1 + beta*L)**alpha [-] of growth; zero
        makes growth size independent, and None leaves the factor out of
        the growth law (three entries).
    solubility : float, optional
        Constant solubility [kg/m**3]. Defaults to ``SOLUBILITY``.

    Returns
    -------
    CrystKinetics
        Kinetics with constant B, G0 and D.
    """
    # Rate first, read by the constant law, then E [J/mol] and order [-].
    growth = (GROWTH_RATE, 0.0, 1.0)
    if alpha is not None:
        growth += (alpha, SIZE_FACTOR_SLOPE)

    return CrystKinetics(
        coeff_solub=[solubility],
        nucl_prim=(NUCLEATION_RATE, 0.0, 0.0),
        growth=growth,
        dissolution=(DISSOLUTION_RATE, 0.0, 1.0),
        custom_mechanisms={"nucl_prim": constant_rate, "growth": constant_rate,
                           "dissolution": constant_rate},
    )


def _build(moments, alpha, mechanism_cls=QMOMPopulationBalance,
           crystal_density=LIQUID_SOLUTE_DENSITY, integrator=None,
           solubility=SOLUBILITY):
    """Build the closed batch crystallizer.

    Parameters
    ----------
    moments : numpy.ndarray
        Initial moments [um**k/m**3 of slurry], shape ``(NUM_MOM,)``.
    alpha : float or None
        Size-factor exponent [-]; see ``_kinetics``.
    mechanism_cls : type, optional
        Population balance class. Defaults to ``QMOMPopulationBalance``.
    crystal_density : float, optional
        Crystal density [kg/m**3]. Defaults to the solute's liquid density.
    integrator : IntegratorBackend, optional
        Backend for a solve. None for solver-free evaluation.
    solubility : float, optional
        Constant solubility [kg/m**3]. Defaults to ``SOLUBILITY``.

    Returns
    -------
    tuple
        The compiled ``BatchCrystallizer``, its population balance and its
        liquid phase.
    """
    liquid = LiquidPhase(DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC,
                         temp=TEMP)
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)
    mechanism = mechanism_cls(
        owning_phase=solid, target_components=TARGET, solvent_name="solvent",
        moments_init=moments, density=crystal_density, kv=SHAPE_FACTOR)
    solid.mechanisms = mechanism

    vessel = BatchCrystallizer(integrator=integrator, isothermal=True)
    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = _kinetics(alpha, solubility)
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
        Liquid species rates [kg/s], the moments [um**k/m**3 of slurry] and
        their rates [um**k/(m**3 s)].
    """
    states = np.asarray(vessel.create_solver_init_states(), dtype=float)
    rates = np.asarray(vessel.unit_model(0.0, states, mat_bce=True), dtype=float).copy()

    collection = vessel.solver_state_collection
    species_key = next(k for k in collection.states if k.name == "mass_j")
    moments_key = next(k for k in collection.states if k.name == "mu_n")

    return (rates[collection.slices[species_key]],
            states[collection.slices[moments_key]],
            rates[collection.slices[moments_key]])


def _population_moments(sizes, counts):
    """Moments of a discrete population.

    Parameters
    ----------
    sizes : tuple of float
        Crystal sizes [um].
    counts : tuple of float
        Crystals at each size [#/m**3 of slurry].

    Returns
    -------
    numpy.ndarray
        mu_0..mu_5 [um**k/m**3], shape ``(NUM_MOM,)``.
    """
    sizes = np.asarray(sizes)
    counts = np.asarray(counts)
    return np.array([np.sum(counts * sizes ** k) for k in range(NUM_MOM)])


def _grown_size(initial_size, age, alpha):
    """Size of a crystal growing along dL/dt = G0 (1 + beta*L)**alpha.

    Parameters
    ----------
    initial_size : float
        Size at age zero [um].
    age : float
        Time grown [s].
    alpha : float
        Size-factor exponent [-].

    Returns
    -------
    float
        Size [um]. Separating variables gives (1 + beta*L)**(1-alpha) =
        (1 + beta*L0)**(1-alpha) + (1-alpha) beta G0 t, and for alpha = 1
        1 + beta*L = (1 + beta*L0) exp(beta G0 t).
    """
    beta = SIZE_FACTOR_SLOPE
    if alpha == 1:
        return ((1 + beta * initial_size) * np.exp(beta * GROWTH_RATE * age) - 1) / beta

    power = 1 - alpha
    return (((1 + beta * initial_size) ** power + power * beta * GROWTH_RATE * age)
            ** (1 / power) - 1) / beta


def _exact_moments(time, alpha, seeded):
    """Moments from the method of characteristics.

    Parameters
    ----------
    time : float
        Time [s].
    alpha : float
        Size-factor exponent [-].
    seeded : bool
        Whether the batch starts with ``SEED_COUNT`` seeds of ``SEED_SIZE``.

    Returns
    -------
    numpy.ndarray
        mu_0..mu_5 [um**k/m**3]: the seeds' ``N L(t)**k`` plus the nuclei's
        ``B * integral over age s of L(s)**k``, born at zero size at a
        constant rate.
    """
    orders = np.arange(NUM_MOM)
    seeds = (SEED_COUNT * _grown_size(SEED_SIZE, time, alpha) ** orders
             if seeded else np.zeros(NUM_MOM))
    nuclei = np.array([
        quad(lambda age: _grown_size(0.0, age, alpha) ** k, 0.0, time,
             epsabs=0.0, epsrel=1e-13, limit=200)[0]
        for k in orders])
    return seeds + NUCLEATION_RATE * nuclei


def _seeds_beside_nuclei_with_integration_error():
    """Seeds at 100 um and nuclei at zero size, m_1 low by CVode's rtol.

    Returns
    -------
    numpy.ndarray
        Moments [um**k/m**3], shape ``(NUM_MOM,)``. The error puts the
        nuclei node below zero, so the quadrature of L**k differs from m_k
        by about 1e-6 there, which only the exact moment form avoids.
    """
    moments = _population_moments((0.0, 100.0), (5e7, 1e10))
    moments[1] *= 1 - CVODE_DEFAULT_RTOL
    return moments


@pytest.mark.parametrize("alpha", [None, 0.0],
                         ids=["three_entry_law", "zero_exponent"])
def test_growth_without_a_size_factor_reduces_exactly_to_the_moment_form(alpha):
    moments = _seeds_beside_nuclei_with_integration_error()

    qmom, _, _ = _build(moments, alpha)
    plain, _, _ = _build(moments, alpha, mechanism_cls=MomentsPopulationBalance)
    qmom_species, _, qmom_rates = _state_and_rates(qmom)
    plain_species, _, plain_rates = _state_and_rates(plain)

    assert -plain_species[TARGET_INDEX] > ACTIVE_MASS_RATE_FLOOR
    np.testing.assert_allclose(qmom_rates, plain_rates, rtol=ALGEBRAIC_RTOL)
    np.testing.assert_allclose(qmom_species, plain_species, rtol=ALGEBRAIC_RTOL)


def test_linear_growth_gives_the_exact_moment_equations():
    # Three sizes, not in size order, so all three nodes carry crystals.
    moments = _population_moments((60.0, 310.0, 5.0), (4e9, 3e7, 1e12))
    vessel, _, _ = _build(moments, LINEAR_EXPONENT)

    _, state, rates = _state_and_rates(vessel)

    # G(L) = G0 (1 + beta*L) turns k G0 integral of L**(k-1) (1 + beta*L) n
    # into k G0 (mu_(k-1) + beta mu_k), with nuclei born at zero size.
    orders = np.arange(1, NUM_MOM)
    expected = np.concatenate((
        [NUCLEATION_RATE],
        orders * GROWTH_RATE * (state[:-1] + SIZE_FACTOR_SLOPE * state[1:])))
    np.testing.assert_allclose(rates, expected, rtol=ALGEBRAIC_RTOL)


def test_dissolution_stays_size_independent_beside_size_dependent_growth():
    # Undersaturated, so only dissolution acts: dmu_k/dt = k D mu_(k-1),
    # whatever the growth law's size factor.
    moments = _population_moments((60.0, 310.0, 5.0), (4e9, 3e7, 1e12))
    vessel, _, _ = _build(moments, SQUARE_ROOT_EXPONENT,
                          solubility=UNDERSATURATED_SOLUBILITY)

    species_rates, state, rates = _state_and_rates(vessel)

    orders = np.arange(1, NUM_MOM)
    expected = np.concatenate(([0.0], orders * DISSOLUTION_RATE * state[:-1]))
    assert species_rates[TARGET_INDEX] > ACTIVE_MASS_RATE_FLOOR
    np.testing.assert_allclose(rates, expected, rtol=ALGEBRAIC_RTOL)


def test_closed_batch_conserves_mass_with_size_dependent_growth():
    """The solid gains exactly the mass the liquid loses, unequal densities.

    The solid mass is ``rho_s * phi * V_L / (1 - phi)``, with the volume
    fraction phi linear in mu_3 and the liquid volume ``sum(m_i / rho_i)``,
    so its rate follows by the chain rule from the published rates alone.
    """
    moments = _population_moments((60.0, 310.0, 5.0), (4e9, 3e7, 1e12))
    vessel, mechanism, liquid = _build(moments, SQUARE_ROOT_EXPONENT,
                                       crystal_density=CRYSTAL_DENSITY)
    species_rates, state, rates = _state_and_rates(vessel)

    liquid_densities, _ = liquid.getDensityPure("liquid")  # [kg/m**3]
    vol_liquid = float(liquid.vol)  # [m**3]
    vol_liquid_rate = float(np.sum(species_rates / liquid_densities))  # [m**3/s]
    phi = mechanism.solid_volume_fraction(state)  # [-]
    phi_rate = mechanism.solid_volume_fraction(rates)  # [1/s]
    solid_mass_rate = CRYSTAL_DENSITY * (phi_rate * vol_liquid / (1.0 - phi) ** 2
                                         + phi / (1.0 - phi) * vol_liquid_rate)  # [kg/s]
    liquid_mass_rate = float(species_rates.sum())  # [kg/s]

    assert -species_rates[TARGET_INDEX] > ACTIVE_MASS_RATE_FLOOR
    assert liquid_mass_rate + solid_mass_rate == pytest.approx(
        0.0, abs=ALGEBRAIC_RTOL * abs(liquid_mass_rate))


@pytest.mark.integration
@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "unseeded"])
@pytest.mark.parametrize("alpha, closure_rtol", [
    (LINEAR_EXPONENT, EXACT_CLOSURE_RTOL),
    (SQUARE_ROOT_EXPONENT, SQUARE_ROOT_CLOSURE_RTOL),
], ids=["linear", "square_root"])
def test_qmom_follows_the_exact_solution_over_a_solve(alpha, closure_rtol, seeded):
    initial = (SEED_COUNT * SEED_SIZE ** np.arange(NUM_MOM) if seeded
               else np.zeros(NUM_MOM))  # [um**k/m**3]
    vessel, _, _ = _build(
        initial, alpha,
        integrator=ScipyBackend(method="LSODA",
                                options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}))

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    moments = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    np.testing.assert_allclose(moments[-1], _exact_moments(RUNTIME, alpha, seeded),
                               rtol=closure_rtol)


def test_qmom_keeps_the_onset_warning_of_the_moment_form():
    # Case-study secondary nucleation, whose s_2 = 1e-5 switches at mu_3 = 0
    # (#529): k_s [#/m**3/s], E [J/mol], s_1 [-], s_2 [-].
    kinetics = CrystKinetics(coeff_solub=[SOLUBILITY],
                             nucl_sec=(4.46e10, 0.0, 2.0, 1e-5),
                             growth=(GROWTH_RATE, 0.0, 1.0))
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    with pytest.warns(RuntimeWarning, match="#529"):
        QMOMPopulationBalance(owning_phase=solid, target_components=TARGET,
                              solvent_name="solvent", moments_init=np.zeros(NUM_MOM),
                              kinetics=kinetics)


@pytest.mark.parametrize("num_mom", [2, 3, 5])
def test_moment_counts_without_a_quadrature_are_refused(num_mom):
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    with pytest.raises(PharmaPyValueError, match="even number of moments"):
        QMOMPopulationBalance(owning_phase=solid, target_components=TARGET,
                              solvent_name="solvent", moments_init=np.zeros(num_mom))
