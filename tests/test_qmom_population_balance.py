"""QMOM population balance in a closed, isothermal batch crystallizer.

Nucleation, growth and dissolution run at constant rates (``CrystKinetics``
custom mechanisms reading their first parameter), so the references are
closed form: the moment equations on one right-hand side, and over a solve
the method of characteristics for seeds and for nuclei born at zero size.
Growth carries the size factor (1 + beta*L)**alpha. Agglomeration runs
alone, with zero growth and nucleation, against collision counting and the
exact zeroth moment of the constant, sum and product kernels. The crystal
density equals the liquid density of the solute, so the slurry volume stays
constant, except in the mass-balance test, which makes them differ.

Units: mass [kg], density [kg/m**3], sizes [um], moments [um**k/m**3 of
slurry], rates per second. The solves use SciPy LSODA, about 2 s in all.
"""

import os

import numpy as np
import pytest
from scipy.integrate import quad

from PharmaPy.CrystallizationKernels import (BrownianKernel, ConstantKernel,
                                             DifferentialSedimentationKernel, KernelConditions,
                                             ProductKernel, ShearKernel, SumKernel)
from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.Errors import (PharmaPySpecificationError, PharmaPyTypeError,
                             PharmaPyValueError)
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
# Linear growth, and mu_0 under the constant, sum and product kernels, are
# exact under QMOM, so only integration error remains: at most 1.4e-10 over
# these solves (3 Nov 2026), a margin of 70.
EXACT_CLOSURE_RTOL = 1e-8  # [-]
# Agglomeration conserves crystal volume, so mu_3 moves only by the rounding
# of the summed cubes; it did not move at all over these solves.
MU3_CONSERVATION_RTOL = 1e-12  # [-]
METRE_PER_MICRON = 1e-6  # [m/um], exact
# A sphere of diameter L has volume (pi/6) L**3; unequal to one, so a lost
# shape factor shows in the sum and product kernels.
SPHERE_SHAPE_FACTOR = np.pi / 6  # [-]
# Three sizes, not in size order, so all three nodes carry crystals.
THREE_SIZES = ((60.0, 310.0, 5.0), (4e9, 3e7, 1e12))  # [um], [#/m**3]
# Narrower, so the product kernel can thin it markedly before it gels.
NARROW_SIZES = ((40.0, 60.0, 80.0), (1e10, 1e10, 1e10))  # [um], [#/m**3]
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


def _kinetics(alpha, solubility=SOLUBILITY, growth_rate=GROWTH_RATE,
              nucleation_rate=NUCLEATION_RATE):
    """Constant-rate nucleation, growth and dissolution.

    Parameters
    ----------
    alpha : float or None
        Exponent of the size factor (1 + beta*L)**alpha [-] of growth; zero
        makes growth size independent, and None leaves the factor out of
        the growth law (three entries).
    solubility : float, optional
        Constant solubility [kg/m**3]. Defaults to ``SOLUBILITY``.
    growth_rate : float, optional
        G0 [um/s]. Defaults to ``GROWTH_RATE``.
    nucleation_rate : float, optional
        B [#/(m**3 s)]. Defaults to ``NUCLEATION_RATE``.

    Returns
    -------
    CrystKinetics
        Kinetics with constant B, G0 and D.
    """
    # Rate first, read by the constant law, then E [J/mol] and order [-].
    growth = (growth_rate, 0.0, 1.0)
    if alpha is not None:
        growth += (alpha, SIZE_FACTOR_SLOPE)

    return CrystKinetics(
        coeff_solub=[solubility],
        nucl_prim=(nucleation_rate, 0.0, 0.0),
        growth=growth,
        dissolution=(DISSOLUTION_RATE, 0.0, 1.0),
        custom_mechanisms={"nucl_prim": constant_rate, "growth": constant_rate,
                           "dissolution": constant_rate},
    )


def _build(moments, alpha, mechanism_cls=QMOMPopulationBalance,
           crystal_density=LIQUID_SOLUTE_DENSITY, integrator=None,
           solubility=SOLUBILITY, shape_factor=SHAPE_FACTOR, agglomeration=None,
           growth_rate=GROWTH_RATE, nucleation_rate=NUCLEATION_RATE, rad=0.0):
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
    shape_factor : float, optional
        Volume shape factor kv [-]. Defaults to ``SHAPE_FACTOR``.
    agglomeration : callable, optional
        Agglomeration kernel for ``QMOMPopulationBalance``; None for none.
    growth_rate, nucleation_rate : float, optional
        G0 [um/s] and B [#/(m**3 s)]; see ``_kinetics``.
    rad : float, optional
        Nucleus size [um]. Defaults to zero.

    Returns
    -------
    tuple
        The compiled ``BatchCrystallizer``, its population balance and its
        liquid phase.
    """
    liquid = LiquidPhase(DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC,
                         temp=TEMP)
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)
    kernel_kwargs = {} if agglomeration is None else {"agglomeration": agglomeration}
    mechanism = mechanism_cls(
        owning_phase=solid, target_components=TARGET, solvent_name="solvent",
        moments_init=moments, density=crystal_density, kv=shape_factor, rad=rad,
        **kernel_kwargs)
    solid.mechanisms = mechanism

    vessel = BatchCrystallizer(integrator=integrator, isothermal=True)
    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = _kinetics(alpha, solubility, growth_rate, nucleation_rate)
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


def _agglomeration_only(moments, kernel, **kwargs):
    """Batch crystallizer in which the crystals only agglomerate.

    Parameters
    ----------
    moments : numpy.ndarray
        Initial moments [um**k/m**3], shape ``(NUM_MOM,)``.
    kernel : callable
        Agglomeration kernel.
    **kwargs
        Passed to ``_build``.

    Returns
    -------
    tuple
        As ``_build``, with nuclei of ``NUCLEUS_SIZE`` unless ``rad`` is
        given. Growth and nucleation are zero, so their transfer carries no
        crystals, but it still runs, in both directions.
    """
    kwargs.setdefault("rad", NUCLEUS_SIZE)
    return _build(moments, None, agglomeration=kernel, growth_rate=0.0,
                  nucleation_rate=0.0, **kwargs)


def _assert_rates_match(rates, expected, scale):
    """Compare moment rates whose expected value can be zero.

    Parameters
    ----------
    rates, expected : numpy.ndarray
        Moment rates [um**k/(m**3 s)], shape ``(NUM_MOM,)``.
    scale : numpy.ndarray
        Gross rate of each moment, the sum of what is gained and lost
        [um**k/(m**3 s)], against which the difference is measured.
    """
    assert np.all(np.abs(rates - expected) <= ALGEBRAIC_RTOL * scale), (rates, expected)


# Viscosity of the order of water's and a moderate density difference; a 1 %
# collision efficiency keeps settling from sweeping up nearly every crystal
# of THREE_SIZES within the solve, as it does at full efficiency.
VISCOSITY = 1e-3  # [Pa s]
DENSITY_DIFFERENCE = 300.0  # [kg/m**3]
# Kernel strengths are otherwise arbitrary.
AGGLOMERATION_KERNELS = {
    "constant": ConstantKernel(1e-14),
    "brownian": BrownianKernel(VISCOSITY),
    "sum": SumKernel(0.5),
    "shear": ShearKernel(20.0, 0.4),
    "sedimentation": DifferentialSedimentationKernel(DENSITY_DIFFERENCE, VISCOSITY, 0.01),
    "product": ProductKernel(1e14),
}
# Anti-vacuity floor for the fraction of crystals a solve removes. Brownian
# collisions, the slowest at these sizes, removed 3.4e-3 (3 Nov 2026).
THINNING_FLOOR = 1e-3  # [-]
# About a nanometre, the order of a critical nucleus; finite, as the Brownian
# kernel needs, and far below every crystal of the agglomeration fixtures.
NUCLEUS_SIZE = 1e-3  # [um]


@pytest.mark.parametrize("name", AGGLOMERATION_KERNELS)
def test_agglomeration_leaves_one_crystal_of_the_summed_volume_per_collision(name):
    # Two sizes, so three kinds of collision, each removing its pair and
    # leaving one crystal of the pair's volume. Zero growth leaves both
    # transfer directions running, which must not count agglomeration twice.
    kernel = AGGLOMERATION_KERNELS[name]
    (small, large), (n_small, n_large) = (20.0, 90.0), (3e10, 2e9)  # [um], [#/m**3]
    moments = _population_moments((small, large), (n_small, n_large))
    vessel, _, _ = _agglomeration_only(moments, kernel, shape_factor=SPHERE_SHAPE_FACTOR)

    _, _, rates = _state_and_rates(vessel)

    # The built-in kernels read only the temperature and the shape factor.
    conditions = KernelConditions(temp=TEMP, supersaturation=0.0, growth_rate=0.0,
                                  shape_factor=SPHERE_SHAPE_FACTOR)

    beta = {pair: float(kernel(np.array(pair[0]), np.array(pair[1]), conditions))
            for pair in ((small, small), (large, large), (small, large))}  # [m**3/s]

    collisions = [  # (collisions per m**3 of slurry per second, sizes removed [um])
        (beta[small, small] * n_small ** 2 / 2, (small, small)),
        (beta[large, large] * n_large ** 2 / 2, (large, large)),
        (beta[small, large] * n_small * n_large, (small, large)),
    ]
    orders = np.arange(NUM_MOM)
    gained = sum(frequency * np.cbrt(sum(size ** 3 for size in pair)) ** orders
                 for frequency, pair in collisions)
    lost = sum(frequency * sum(size ** orders for size in pair)
               for frequency, pair in collisions)
    _assert_rates_match(rates, gained - lost, gained + lost)


# Each identity holds at any kernel strength; the constants are arbitrary.
@pytest.mark.parametrize("kernel, zeroth_moment_rate", [
    (ConstantKernel(1e-14), lambda mu, kv: -1e-14 * mu[0] ** 2 / 2),
    (SumKernel(0.5), lambda mu, kv: -0.5 * mu[0] * kv * mu[3] * METRE_PER_MICRON ** 3),
    (ProductKernel(1e14), lambda mu, kv: -1e14 * (kv * mu[3] * METRE_PER_MICRON ** 3) ** 2 / 2),
    # 1/2 sum_ij w_i w_j (L_i + L_j)**3 expands to mu_0 mu_3 + 3 mu_1 mu_2.
    (ShearKernel(20.0, 0.4), lambda mu, kv: -(0.4 * 20.0 / 6) * METRE_PER_MICRON ** 3
     * (mu[0] * mu[3] + 3 * mu[1] * mu[2])),
], ids=["constant", "sum", "product", "shear"])
def test_built_in_kernels_give_the_exact_zeroth_moment_rate(kernel, zeroth_moment_rate):
    moments = _population_moments(*THREE_SIZES)
    vessel, _, _ = _agglomeration_only(moments, kernel, shape_factor=SPHERE_SHAPE_FACTOR)

    _, state, rates = _state_and_rates(vessel)

    expected = zeroth_moment_rate(state, SPHERE_SHAPE_FACTOR)  # [#/(m**3 s)]
    assert rates[0] == pytest.approx(expected, rel=ALGEBRAIC_RTOL)
    # Volume is conserved: each collision moves the mean crystal volume.
    assert rates[3] == pytest.approx(
        0.0, abs=ALGEBRAIC_RTOL * abs(expected) * state[3] / state[0])


def _agglomeration_case(name):
    """Initial moments, kernel, runtime and exact mu_0 of an agglomeration solve.

    Parameters
    ----------
    name : {'constant', 'sum', 'product'}
        Kernel.

    Returns
    -------
    tuple
        Moments [um**k/m**3], the kernel, the runtime [s] and mu_0 at its end
        [#/m**3] from the exact solution (Scott, 1968): mu_0 / (1 + beta_0
        mu_0 t / 2), mu_0 exp(-beta_0 phi t), and mu_0 - beta_0 phi**2 t / 2
        before gelation, with phi = kv mu_3 the crystal volume fraction.
    """
    if name == "product":
        sizes, counts = NARROW_SIZES
        moments = _population_moments(sizes, counts)
        rate_constant = 1e14  # [1/(m**3 s)]
        volumes = SHAPE_FACTOR * (np.asarray(sizes) * METRE_PER_MICRON) ** 3  # [m**3]
        # The population gels at 1 / (beta_0 sum n v**2); stop at half of it.
        runtime = 0.5 / (rate_constant * np.sum(np.asarray(counts) * volumes ** 2))  # [s]
        phi = SHAPE_FACTOR * moments[3] * METRE_PER_MICRON ** 3  # [-]
        return (moments, ProductKernel(rate_constant), runtime,
                moments[0] - rate_constant * phi ** 2 * runtime / 2)

    moments = _population_moments(*THREE_SIZES)
    phi = SHAPE_FACTOR * moments[3] * METRE_PER_MICRON ** 3  # [-]
    if name == "constant":
        rate_constant = 2 / (moments[0] * RUNTIME)  # [m**3/s]: mu_0 halves
        return (moments, ConstantKernel(rate_constant), RUNTIME,
                moments[0] / (1 + rate_constant * moments[0] * RUNTIME / 2))

    rate_constant = 1 / (phi * RUNTIME)  # [1/s]: mu_0 falls by e
    return (moments, SumKernel(rate_constant), RUNTIME,
            moments[0] * np.exp(-rate_constant * phi * RUNTIME))


@pytest.mark.integration
@pytest.mark.parametrize("name", ["constant", "sum", "product"])
def test_agglomeration_follows_the_exact_zeroth_moment_over_a_solve(name):
    moments, kernel, runtime, expected_mu0 = _agglomeration_case(name)
    vessel, _, _ = _agglomeration_only(
        moments, kernel,
        integrator=ScipyBackend(method="LSODA",
                                options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}))

    vessel.solve_unit(runtime=runtime, verbose=False)

    trajectory = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    assert trajectory[-1, 0] == pytest.approx(expected_mu0, rel=EXACT_CLOSURE_RTOL)
    np.testing.assert_allclose(trajectory[:, 3], moments[3], rtol=MU3_CONSERVATION_RTOL)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["shear", "brownian", "sedimentation"])
def test_physical_kernels_thin_the_population_at_constant_volume_over_a_solve(name):
    # No closed form here, so the solve must keep the crystal volume, only
    # ever lose crystals, and lose a measurable share of them.
    moments = _population_moments(*THREE_SIZES)
    vessel, _, _ = _agglomeration_only(
        moments, AGGLOMERATION_KERNELS[name], shape_factor=SPHERE_SHAPE_FACTOR,
        integrator=ScipyBackend(method="LSODA",
                                options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}))

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    trajectory = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    assert np.all(np.isfinite(trajectory))
    np.testing.assert_allclose(trajectory[:, 3], moments[3], rtol=MU3_CONSERVATION_RTOL)
    assert np.all(np.diff(trajectory[:, 0]) <= 0)
    assert 1 - trajectory[-1, 0] / moments[0] > THINNING_FLOOR


class GrowthPacedKernel:
    """Agglomeration proportional to the growth rate, keeping the conditions it saw.

    Parameters
    ----------
    coefficient : float
        beta / G0 [m**3/um].
    """

    def __init__(self, coefficient):
        self.coefficient = coefficient  # [m**3/um]
        self.seen = []

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um].
        conditions : KernelConditions
            Recorded, and its growth rate used.

        Returns
        -------
        numpy.ndarray
            coefficient * G0 [m**3/s] for every pair.
        """
        self.seen.append(conditions)
        return np.full(np.broadcast(size_i, size_j).shape,
                       self.coefficient * conditions.growth_rate)


def test_a_user_kernel_sees_the_state_it_acts_on():
    kernel = GrowthPacedKernel(1e-13)
    moments = _population_moments(*THREE_SIZES)
    vessel, _, liquid = _build(moments, None, agglomeration=kernel,
                               shape_factor=SPHERE_SHAPE_FACTOR)

    _, state, rates = _state_and_rates(vessel)

    # Relative supersaturation on the default basis, kg of C per m**3 of
    # solution, against the constant solubility.
    concentration = CHARGE_MASS * CHARGE_MASS_FRAC[TARGET_INDEX] / float(liquid.vol)  # [kg/m**3]
    conditions = kernel.seen[-1]
    assert conditions.temp == pytest.approx(TEMP, rel=ALGEBRAIC_RTOL)
    assert conditions.supersaturation == pytest.approx(
        (concentration - SOLUBILITY) / SOLUBILITY, rel=ALGEBRAIC_RTOL)
    assert conditions.growth_rate == pytest.approx(GROWTH_RATE, rel=ALGEBRAIC_RTOL)
    assert conditions.shape_factor == SPHERE_SHAPE_FACTOR
    # Nucleation adds crystals at zero size, agglomeration removes them.
    assert rates[0] == pytest.approx(
        NUCLEATION_RATE - 1e-13 * GROWTH_RATE * state[0] ** 2 / 2, rel=ALGEBRAIC_RTOL)


@pytest.mark.parametrize("kernel", [
    lambda size_i, size_j, conditions: np.full(np.broadcast(size_i, size_j).shape, -1e-14),
    lambda size_i, size_j, conditions: np.full(np.broadcast(size_i, size_j).shape, np.nan),
    lambda size_i, size_j, conditions: np.ones(3),
], ids=["negative", "nan", "unbroadcast"])
def test_kernel_values_that_are_not_rates_are_refused(kernel):
    vessel, _, _ = _agglomeration_only(_population_moments(*THREE_SIZES), kernel)

    with pytest.raises(PharmaPyValueError, match="agglomeration kernel must"):
        _state_and_rates(vessel)


def test_brownian_agglomeration_needs_nuclei_of_finite_size():
    # Nuclei born at rad = 0 would sit where the Brownian kernel is infinite.
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    with pytest.raises(PharmaPyValueError, match="rad > 0"):
        QMOMPopulationBalance(owning_phase=solid, target_components=TARGET,
                              solvent_name="solvent", moments_init=np.zeros(NUM_MOM),
                              agglomeration=BrownianKernel(VISCOSITY))


class SizeRecordingKernel:
    """Constant agglomeration that keeps the smallest size it was asked about."""

    def __init__(self):
        self.smallest = np.inf  # [um]

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um], recorded.
        conditions : KernelConditions
            Unused.

        Returns
        -------
        numpy.ndarray
            1e-14 [m**3/s], an arbitrary constant, for every pair.
        """
        self.smallest = min(self.smallest, float(np.min(size_i)), float(np.min(size_j)))
        return np.full(np.broadcast(size_i, size_j).shape, 1e-14)


def test_kernels_see_no_crystal_smaller_than_a_nucleus():
    # Rounding puts the node of nuclei still at zero size at about 5e-13 um,
    # far below a nucleus; a 1/L kernel would make its rate absurd.
    kernel = SizeRecordingKernel()
    moments = _population_moments((0.0, 100.0), (5e7, 1e10))
    vessel, _, _ = _agglomeration_only(moments, kernel)

    _state_and_rates(vessel)

    assert kernel.smallest == NUCLEUS_SIZE


def test_an_agglomeration_kernel_must_be_callable():
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    with pytest.raises(PharmaPyTypeError, match="kernel callable"):
        QMOMPopulationBalance(owning_phase=solid, target_components=TARGET,
                              solvent_name="solvent", moments_init=np.zeros(NUM_MOM),
                              agglomeration=1e-14)


def test_agglomeration_left_out_of_the_vessel_is_refused():
    vessel, _, _ = _agglomeration_only(_population_moments(*THREE_SIZES),
                                       ConstantKernel(1e-14))

    vessel.intraphase_processes = []

    with pytest.raises(PharmaPySpecificationError, match="intraphase_processes"):
        _state_and_rates(vessel)
