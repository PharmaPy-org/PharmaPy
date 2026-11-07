"""QMOM population balance in isothermal batch, semibatch and continuous crystallizers.

Nucleation, growth and dissolution run at constant rates (``CrystKinetics``
custom mechanisms reading their first parameter), so the references are
closed form: the moment equations on one right-hand side, and over a solve
the method of characteristics for seeds and for nuclei born at zero size,
or the steady MSMPR moments. Growth carries the size factor
(1 + beta*L)**alpha. Agglomeration runs alone, with zero growth and
nucleation, against collision counting and the exact zeroth moment of the
constant, sum and product kernels. The crystal density equals the liquid
density of the solute, so the slurry volume stays constant, except in the
mass-balance tests, which make them differ. Flowsheet tests connect real
units through ``SimulationExec``.

Units: mass [kg], density [kg/m**3], sizes [um], moments [um**k/m**3 of
slurry], rates per second. The solves use SciPy LSODA, about 30 s in all.
"""

import os

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from PharmaPy.Connections import Connection
from PharmaPy.CrystallizationKernels import (BinaryFragmentation, BrownianKernel, ConstantKernel,
                                             ConstantSelection, DifferentialSedimentationKernel,
                                             ErosionFragmentation, KernelConditions,
                                             ParabolicFragmentation, PowerLawSelection,
                                             ProductKernel, ShearKernel, SumKernel,
                                             UniformFragmentation)
from PharmaPy.Crystallizers_Refactored import (BatchCrystallizer, ContinuousCrystallizer,
                                               SemiBatchCrystallizer)
from PharmaPy.DataClasses import StateKey
from PharmaPy.Errors import (PharmaPySpecificationError, PharmaPyTypeError,
                             PharmaPyValueError)
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import (MomentsPopulationBalance, OneDFVMMechanism,
                                 QMOMPopulationBalance)
from PharmaPy.MixedPhases import Slurry as OldSlurry
from PharmaPy.Phases import LiquidPhase as OldLiquidPhase
from PharmaPy.Phases import SolidPhase as OldSolidPhase
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import SolidStream as OldSolidStream
from PharmaPy.Streams_Refactored import LiquidStream

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
# Two sizes, for a two-node rule.
FEW_SIZES = ((20.0, 90.0), (3e10, 2e9))  # [um], [#/m**3]
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
           growth_rate=GROWTH_RATE, nucleation_rate=NUCLEATION_RATE, rad=0.0,
           breakage=None, fragments=None, vessel_cls=BatchCrystallizer,
           feed_mass_flow=None):
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
    breakage, fragments : optional
        Breakage selection function and fragment distribution for
        ``QMOMPopulationBalance``; None for none.
    vessel_cls : type, optional
        Crystallizer class. Defaults to ``BatchCrystallizer``.
    feed_mass_flow : float, optional
        Mass flow of a crystal-free feed of the charge's own liquid [kg/s];
        None for no feed.

    Returns
    -------
    tuple
        The compiled crystallizer, its population balance and its liquid
        phase.
    """
    liquid = LiquidPhase(DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC,
                         temp=TEMP, name_solv="solvent")
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)
    kernel_kwargs = {name: law for name, law in (("agglomeration", agglomeration),
                                                 ("breakage", breakage),
                                                 ("fragments", fragments))
                     if law is not None}
    mechanism = mechanism_cls(
        owning_phase=solid, target_components=TARGET, solvent_name="solvent",
        moments_init=moments, density=crystal_density, kv=shape_factor, rad=rad,
        **kernel_kwargs)
    solid.mechanisms = mechanism

    vessel = vessel_cls(integrator=integrator, isothermal=True)
    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = _kinetics(alpha, solubility, growth_rate, nucleation_rate)
    if feed_mass_flow is not None:
        vessel.Inlet = LiquidStream(DATA_PATH, mass_flow=feed_mass_flow,
                                    mass_frac=CHARGE_MASS_FRAC, temp=TEMP,
                                    name_solv="solvent")
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


@pytest.mark.parametrize("kernels", [
    {"agglomeration": ConstantKernel(1e-14)},
    {"breakage": ConstantSelection(1e-3), "fragments": BinaryFragmentation()},
], ids=["agglomeration", "breakage"])
def test_kernels_left_out_of_the_vessel_are_refused(kernels):
    vessel, _, _ = _build(_population_moments(*THREE_SIZES), None, growth_rate=0.0,
                          nucleation_rate=0.0, **kernels)

    vessel.intraphase_processes = []

    with pytest.raises(PharmaPySpecificationError, match="intraphase_processes"):
        _state_and_rates(vessel)


def _breakage_only(moments, selection, fragments, **kwargs):
    """Batch crystallizer in which the crystals only break.

    Parameters
    ----------
    moments : numpy.ndarray
        Initial moments [um**k/m**3], shape ``(NUM_MOM,)``.
    selection : callable
        Breakage selection function.
    fragments : object
        Fragment distribution.
    **kwargs
        Passed to ``_build``.

    Returns
    -------
    tuple
        As ``_build``, with zero growth and nucleation.
    """
    return _build(moments, None, breakage=selection, fragments=fragments, growth_rate=0.0,
                  nucleation_rate=0.0, **kwargs)


ORDERS = np.arange(NUM_MOM)  # [-]
BREAKAGE_RATE = 1e-3  # S_0 [1/s], arbitrary
# Fragment shares b_k / L**k of the self-similar distributions, as Table 3 of
# Marchisio, Vigil and Fox (2003) gives them, for the same orders.
SELF_SIMILAR_FRAGMENTS = {
    "symmetric": (BinaryFragmentation(0.5), 2 ** ((3 - ORDERS) / 3)),
    "mass_ratio_1_4": (BinaryFragmentation(0.2), (4 ** (ORDERS / 3) + 1) / 5 ** (ORDERS / 3)),
    "parabolic": (ParabolicFragmentation(1.0),
                  3 / (3 + ORDERS) + 0.5 * (72 / (9 + ORDERS) - 72 / (6 + ORDERS)
                                            + 18 / (3 + ORDERS))),
    "uniform": (UniformFragmentation(), 6 / (ORDERS + 3)),
}


@pytest.mark.parametrize("name", SELF_SIMILAR_FRAGMENTS)
def test_constant_breakage_scales_each_moment_by_its_fragment_share(name):
    # Every parent breaks at S_0 into fragments whose k-th moment is c_k L**k,
    # so dmu_k/dt = S_0 (c_k - 1) mu_k for every k.
    fragments, share = SELF_SIMILAR_FRAGMENTS[name]
    moments = _population_moments(*THREE_SIZES)
    vessel, _, _ = _breakage_only(moments, ConstantSelection(BREAKAGE_RATE), fragments)

    _, state, rates = _state_and_rates(vessel)

    expected = BREAKAGE_RATE * (share - 1) * state
    _assert_rates_match(rates, expected, BREAKAGE_RATE * (share + 1) * state)


def test_volume_proportional_breakage_is_exact_for_the_low_moments():
    # a = S_0 (L / L_ref)**3 with symmetric fragments: dmu_k/dt =
    # S_0 (2**(1 - k/3) - 1) mu_(k+3) / L_ref**3, closed for k <= 2.
    reference_size = 50.0  # [um]
    moments = _population_moments(*THREE_SIZES)
    vessel, _, _ = _breakage_only(
        moments, PowerLawSelection(BREAKAGE_RATE, 3.0, reference_size), BinaryFragmentation())

    _, state, rates = _state_and_rates(vessel)

    low = np.arange(3)
    expected = BREAKAGE_RATE * (2 ** (1 - low / 3) - 1) * state[low + 3] / reference_size ** 3
    np.testing.assert_allclose(rates[low], expected, rtol=ALGEBRAIC_RTOL)
    assert rates[3] == pytest.approx(0.0, abs=ALGEBRAIC_RTOL * abs(rates[0]) * state[3] / state[0])


def test_erosion_chips_one_fragment_per_parent_larger_than_it():
    # Parents of 3 um are smaller than the 5 um fragment and stay whole;
    # each 90 um parent loses one 5 um fragment at the constant frequency.
    fragment_size = 5.0  # [um]
    (small, large), (n_small, n_large) = (3.0, 90.0), (3e10, 2e9)  # [um], [#/m**3]
    moments = _population_moments((small, large), (n_small, n_large))
    vessel, _, _ = _breakage_only(moments, ConstantSelection(BREAKAGE_RATE),
                                  ErosionFragmentation(fragment_size))

    _, _, rates = _state_and_rates(vessel)

    events = BREAKAGE_RATE * n_large  # [#/(m**3 s)]
    gained = events * (fragment_size ** ORDERS
                       + np.cbrt(large ** 3 - fragment_size ** 3) ** ORDERS)
    lost = events * large ** ORDERS
    _assert_rates_match(rates, gained - lost, gained + lost)


@pytest.mark.integration
def test_constant_symmetric_breakage_follows_the_exact_moments_over_a_solve():
    # dmu_k/dt = S_0 (2**(1 - k/3) - 1) mu_k for every k; S_0 doubles mu_0.
    rate_constant = np.log(2) / RUNTIME  # [1/s]
    moments = _population_moments(*THREE_SIZES)
    vessel, _, _ = _breakage_only(
        moments, ConstantSelection(rate_constant), BinaryFragmentation(),
        integrator=ScipyBackend(method="LSODA",
                                options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}))

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    trajectory = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    expected = moments * np.exp(rate_constant * (2 ** (1 - ORDERS / 3) - 1) * RUNTIME)
    np.testing.assert_allclose(trajectory[-1], expected, rtol=EXACT_CLOSURE_RTOL)
    np.testing.assert_allclose(trajectory[:, 3], moments[3], rtol=MU3_CONSERVATION_RTOL)


@pytest.mark.integration
def test_volume_proportional_breakage_makes_crystals_at_a_constant_rate_over_a_solve():
    # a = S_0 L**3 / L_ref**3 and two fragments per event: dmu_0/dt =
    # S_0 mu_3 / L_ref**3 at constant mu_3, so mu_0 grows linearly (Ziff and
    # McGrady, 1985). S_0 adds half the initial count.
    reference_size = 50.0  # [um]
    moments = _population_moments(*THREE_SIZES)
    rate_constant = 0.5 * moments[0] * reference_size ** 3 / (moments[3] * RUNTIME)  # [1/s]
    vessel, _, _ = _breakage_only(
        moments, PowerLawSelection(rate_constant, 3.0, reference_size), BinaryFragmentation(),
        integrator=ScipyBackend(method="LSODA",
                                options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}))

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    trajectory = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    assert trajectory[-1, 0] == pytest.approx(1.5 * moments[0], rel=EXACT_CLOSURE_RTOL)
    np.testing.assert_allclose(trajectory[:, 3], moments[3], rtol=MU3_CONSERVATION_RTOL)


@pytest.mark.integration
def test_fragments_below_the_minimum_size_stop_breaking_over_a_solve():
    # Seeds of one size break at S_0 into two fragments smaller than the minimum
    # size, which stay whole. With x = exp(-S_0 t) of the seeds left, mu_k =
    # N (x L**k + 2 (1 - x) (L / 2**(1/3))**k), so the count stops short of 2 N.
    # Without the minimum it would grow as exp(S_0 t), about 148 N here.
    fragment_size = SEED_SIZE / 2 ** (1 / 3)  # [um]
    minimum_size = (fragment_size + SEED_SIZE) / 2  # [um], between the two sizes
    rate_constant = 5 / RUNTIME  # [1/s]: five e-folds of the seeds
    moments = SEED_COUNT * SEED_SIZE ** ORDERS  # [um**k/m**3]
    vessel, _, _ = _breakage_only(
        moments, ConstantSelection(rate_constant, minimum_size=minimum_size), BinaryFragmentation(),
        integrator=ScipyBackend(method="LSODA",
                                options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}))

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    left = np.exp(-rate_constant * np.ravel(vessel.result.time))[:, None]  # [-]
    expected = SEED_COUNT * (left * SEED_SIZE ** ORDERS
                             + 2 * (1 - left) * fragment_size ** ORDERS)  # [um**k/m**3]
    trajectory = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    np.testing.assert_allclose(trajectory, expected, rtol=EXACT_CLOSURE_RTOL)


@pytest.mark.integration
def test_agglomeration_and_breakage_reach_their_balance_over_a_solve():
    # dmu_0/dt = -beta_0 mu_0**2 / 2 + S_0 mu_0 is logistic: mu_0 rises to
    # K = 2 S_0 / beta_0 as K / (1 + (K / mu_0(0) - 1) exp(-S_0 t)).
    moments = _population_moments(*THREE_SIZES)
    rate_constant = 5 / RUNTIME  # [1/s]: five e-folds, close to the balance
    balance = 3 * moments[0]  # [#/m**3]
    vessel, _, _ = _build(
        moments, None, growth_rate=0.0, nucleation_rate=0.0,
        agglomeration=ConstantKernel(2 * rate_constant / balance),
        breakage=ConstantSelection(rate_constant), fragments=BinaryFragmentation(),
        integrator=ScipyBackend(method="LSODA",
                                options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}))

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    trajectory = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    expected_mu0 = balance / (1 + (balance / moments[0] - 1) * np.exp(-rate_constant * RUNTIME))
    assert trajectory[-1, 0] == pytest.approx(expected_mu0, rel=EXACT_CLOSURE_RTOL)
    np.testing.assert_allclose(trajectory[:, 3], moments[3], rtol=MU3_CONSERVATION_RTOL)


class VolumeLosingFragments:
    """Two fragments of half the parent's size, which hold only a quarter of its volume."""

    def moments(self, parent_sizes, orders):
        """Moments of the two fragments.

        Parameters
        ----------
        parent_sizes : numpy.ndarray
            Parent sizes [um], shape ``(n,)``.
        orders : numpy.ndarray
            Moment orders, shape ``(m,)``.

        Returns
        -------
        numpy.ndarray
            2 (L/2)**k [um**k], shape ``(n, m)``.
        """
        return 2 * (np.asarray(parent_sizes)[:, None] / 2) ** np.asarray(orders)[None, :]


@pytest.mark.parametrize("selection, fragments, message", [
    (lambda size, conditions: -np.ones(np.shape(size)), BinaryFragmentation(),
     "selection function must return finite, non-negative"),
    (lambda size, conditions: np.ones(2), BinaryFragmentation(),
     "selection function must return one frequency per size"),
    (ConstantSelection(BREAKAGE_RATE), VolumeLosingFragments(), "hold their parent's volume"),
], ids=["negative_frequency", "unbroadcast_frequency", "volume_losing_fragments"])
def test_breakage_laws_that_break_the_balance_are_refused(selection, fragments, message):
    vessel, _, _ = _breakage_only(_population_moments(*THREE_SIZES), selection, fragments)

    with pytest.raises(PharmaPyValueError, match=message):
        _state_and_rates(vessel)


class ShatteringSelection:
    """A user selection function that is infinite at zero size, a = S_0 / L."""

    needs_finite_nuclei = True

    def __call__(self, size, conditions):
        """Breakage frequency.

        Parameters
        ----------
        size : numpy.ndarray
            Crystal sizes [um].
        conditions : KernelConditions
            Unused.

        Returns
        -------
        numpy.ndarray
            1e-3 / L [1/s], with L in um.
        """
        return 1e-3 / np.asarray(size, dtype=float)


class SizeRecordingSelection:
    """Constant breakage frequency that keeps the smallest size it was asked about."""

    def __init__(self):
        self.smallest = np.inf  # [um]

    def __call__(self, size, conditions):
        """Breakage frequency.

        Parameters
        ----------
        size : numpy.ndarray
            Crystal sizes [um], recorded.
        conditions : KernelConditions
            Unused.

        Returns
        -------
        numpy.ndarray
            ``BREAKAGE_RATE`` [1/s] for every size.
        """
        self.smallest = min(self.smallest, float(np.min(size)))
        return np.full(np.shape(size), BREAKAGE_RATE)


def test_breakage_sees_no_crystal_smaller_than_a_nucleus():
    # As for agglomeration: the node of nuclei still at zero size sits at
    # about 5e-13 um, and the selection function sees the nucleus size.
    selection = SizeRecordingSelection()
    moments = _population_moments((0.0, 100.0), (5e7, 1e10))
    vessel, _, _ = _breakage_only(moments, selection, BinaryFragmentation(), rad=NUCLEUS_SIZE)

    _state_and_rates(vessel)

    assert selection.smallest == NUCLEUS_SIZE


@pytest.mark.parametrize("kernels, error, message", [
    ({"breakage": ConstantSelection(BREAKAGE_RATE)}, PharmaPyValueError, "both a selection"),
    ({"fragments": BinaryFragmentation()}, PharmaPyValueError, "both a selection"),
    ({"breakage": BREAKAGE_RATE, "fragments": BinaryFragmentation()}, PharmaPyTypeError,
     "selection function a"),
    ({"breakage": ConstantSelection(BREAKAGE_RATE), "fragments": 0.5}, PharmaPyTypeError,
     "moments"),
    ({"breakage": ShatteringSelection(), "fragments": BinaryFragmentation()}, PharmaPyValueError,
     "rad > 0"),
], ids=["selection_alone", "fragments_alone", "uncallable_selection", "fragments_without_moments",
        "infinite_at_zero_size"])
def test_incomplete_or_invalid_breakage_is_refused(kernels, error, message):
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    with pytest.raises(error, match=message):
        QMOMPopulationBalance(owning_phase=solid, target_components=TARGET,
                              solvent_name="solvent", moments_init=np.zeros(NUM_MOM),
                              **kernels)


@pytest.mark.parametrize("initial_alpha", [0.0, SQUARE_ROOT_EXPONENT],
                         ids=["from_size_independent", "from_another_exponent"])
def test_qmom_follows_a_size_factor_changed_through_set_params(initial_alpha):
    # Evaluated once, then made linear: the rates must follow the new law,
    # k G0 (mu_(k-1) + beta mu_k), whatever the first evaluation saw.
    moments = _population_moments(*THREE_SIZES)
    vessel, mechanism, _ = _build(moments, initial_alpha)
    _state_and_rates(vessel)
    kinetics = mechanism.mechanism_kinetics
    updated = kinetics.concat_params().copy()
    updated[kinetics.name_params.index("alpha")] = LINEAR_EXPONENT
    kinetics.set_params(updated)

    _, state, rates = _state_and_rates(vessel)

    orders = np.arange(1, NUM_MOM)
    expected = np.concatenate((
        [NUCLEATION_RATE],
        orders * GROWTH_RATE * (state[:-1] + SIZE_FACTOR_SLOPE * state[1:])))
    np.testing.assert_allclose(rates, expected, rtol=ALGEBRAIC_RTOL)


# Residence time of about 100 s for the 1 kg charge.
FAST_FEED = 0.01  # [kg/s]
# Slow enough that a good part of the crystals stays in the vessel for 600 s.
SLOW_FEED = 1e-3  # [kg/s]
# With a 100 s residence time, k beta G0 tau is at most 0.2 up to k = 5, so the
# steady moments of linear growth stay finite.
SLOW_GROWTH_RATE = 0.02  # G0 [um/s]
# 40 residence times: the slowest transient, that of mu_5, is about 4e-12.
STEADY_RUNTIME = 4000.0  # [s]
# Over the flow solves the worst miss was 7.8e-9, on the steady moments of
# linear growth (6 Nov 2026); a margin of ten.
FLOW_SOLVE_RTOL = 1e-7  # [-]
# Each state carries the solver's 1e-10 relative error; a decade of margin.
# The worst miss was 7e-11, on the crystal volume under dilution.
CONSERVATION_RTOL = 1e-9  # [-]


def _lsoda():
    """SciPy LSODA at the module's tight tolerance.

    Returns
    -------
    ScipyBackend
        Backend for one solve.
    """
    return ScipyBackend(method="LSODA", options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL})


def _residence_time(vessel):
    """Slurry residence time at the end of a solve of a continuous vessel.

    Parameters
    ----------
    vessel : ContinuousCrystallizer
        Solved crystallizer.

    Returns
    -------
    float
        Slurry volume over outlet volumetric flow [s].
    """
    volume = np.ravel(vessel.result.vessel_vol)  # [m**3]
    outlet_flow = np.ravel(vessel.result.outlet_vol_flow)  # [m**3/s]
    return volume[-1] / outlet_flow[-1]


@pytest.mark.integration
@pytest.mark.parametrize("alpha, slope", [
    (None, 0.0), (LINEAR_EXPONENT, SIZE_FACTOR_SLOPE),
], ids=["size_independent", "linear_growth"])
def test_continuous_crystallizer_reaches_the_steady_msmpr_moments(alpha, slope):
    # Steady, 0 = k G0 (mu_(k-1) + beta mu_k) - mu_k / tau, with mu_0 = B tau.
    vessel, _, _ = _build(np.zeros(NUM_MOM), alpha, vessel_cls=ContinuousCrystallizer,
                          feed_mass_flow=FAST_FEED, growth_rate=SLOW_GROWTH_RATE,
                          integrator=_lsoda())

    vessel.solve_unit(runtime=STEADY_RUNTIME, verbose=False)

    tau = _residence_time(vessel)
    expected = [NUCLEATION_RATE * tau]
    for k in range(1, NUM_MOM):
        expected.append(k * SLOW_GROWTH_RATE * tau * expected[-1]
                        / (1 - k * slope * SLOW_GROWTH_RATE * tau))
    np.testing.assert_allclose(np.asarray(vessel.result.mu_n_solid0)[-1], expected,
                               rtol=FLOW_SOLVE_RTOL)


@pytest.mark.integration
def test_continuous_crystallizer_settles_at_the_balance_of_both_kernels():
    # Steady, 0 = B - beta_0 mu_0**2 / 2 + S_0 mu_0 - mu_0 / tau: symmetric
    # fragments add one crystal per event, agglomeration removes one.
    selection_rate, kernel_constant = 5e-3, 1e-12  # [1/s], [m**3/s]
    vessel, _, _ = _build(np.zeros(NUM_MOM), None, vessel_cls=ContinuousCrystallizer,
                          feed_mass_flow=FAST_FEED, growth_rate=SLOW_GROWTH_RATE,
                          agglomeration=ConstantKernel(kernel_constant),
                          breakage=ConstantSelection(selection_rate),
                          fragments=BinaryFragmentation(), integrator=_lsoda())

    vessel.solve_unit(runtime=STEADY_RUNTIME, verbose=False)

    net_loss = 1 / _residence_time(vessel) - selection_rate  # [1/s]
    expected = ((np.sqrt(net_loss ** 2 + 2 * kernel_constant * NUCLEATION_RATE) - net_loss)
                / kernel_constant)  # [#/m**3]
    assert np.asarray(vessel.result.mu_n_solid0)[-1, 0] == pytest.approx(
        expected, rel=FLOW_SOLVE_RTOL)


@pytest.mark.integration
def test_semibatch_crystallizer_conserves_mass_with_every_mechanism_on():
    moments = _population_moments(*THREE_SIZES)
    vessel, _, _ = _build(moments, SQUARE_ROOT_EXPONENT, vessel_cls=SemiBatchCrystallizer,
                          feed_mass_flow=SLOW_FEED, growth_rate=SLOW_GROWTH_RATE,
                          crystal_density=CRYSTAL_DENSITY,
                          agglomeration=ConstantKernel(1e-14),
                          breakage=ConstantSelection(1e-3),
                          fragments=BinaryFragmentation(), integrator=_lsoda())

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    result = vessel.result
    time = np.ravel(result.time)  # [s]
    liquid_mass = np.asarray(result.mass_j_liquid0).sum(axis=1)  # [kg]
    solid_mass = (CRYSTAL_DENSITY * SHAPE_FACTOR * np.asarray(result.mu_n_solid0)[:, 3]
                  * METRE_PER_MICRON ** 3 * np.ravel(result.vessel_vol))  # [kg]
    total_mass = liquid_mass + solid_mass  # [kg]

    # Guard against a vacuous pass: the crystals did grow.
    assert solid_mass[-1] > solid_mass[0]
    np.testing.assert_allclose(total_mass, total_mass[0] + SLOW_FEED * time,
                               rtol=CONSERVATION_RTOL)
    np.testing.assert_allclose(np.ravel(result.Total_m_in_vessel), total_mass,
                               rtol=CONSERVATION_RTOL)


@pytest.mark.integration
def test_semibatch_feed_dilutes_the_kernels_at_constant_crystal_volume():
    # With no growth and a crystal-free feed, the kernels alone change the
    # count N = mu_0 V: dN/dt = V (-beta_0 (N/V)**2 / 2 + S_0 N/V), V growing
    # at the feed's volumetric flow, while the crystal volume mu_3 V stays put.
    kernel_constant, selection_rate = 2e-15, 1e-3  # [m**3/s], [1/s]
    moments = _population_moments(*THREE_SIZES)
    vessel, _, _ = _build(moments, None, vessel_cls=SemiBatchCrystallizer,
                          feed_mass_flow=SLOW_FEED, growth_rate=0.0, nucleation_rate=0.0,
                          agglomeration=ConstantKernel(kernel_constant),
                          breakage=ConstantSelection(selection_rate),
                          fragments=BinaryFragmentation(), integrator=_lsoda())
    feed_flow = float(LiquidStream(DATA_PATH, mass_flow=SLOW_FEED, mass_frac=CHARGE_MASS_FRAC,
                                   temp=TEMP, name_solv="solvent").vol_flow)  # [m**3/s]

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    trajectory = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    volume = np.ravel(vessel.result.vessel_vol)  # [m**3]

    def count_rate(time, count):
        """dN/dt [#/s] at time [s] and count N [#]."""
        held = volume[0] + feed_flow * time  # [m**3]
        return held * (-kernel_constant * (count / held) ** 2 / 2
                       + selection_rate * count / held)

    reference = solve_ivp(count_rate, (0.0, RUNTIME), [moments[0] * volume[0]],
                          method="LSODA", rtol=1e-12, atol=1.0)
    assert trajectory[-1, 0] * volume[-1] == pytest.approx(reference.y[0, -1],
                                                           rel=FLOW_SOLVE_RTOL)
    np.testing.assert_allclose(trajectory[:, 3] * volume, moments[3] * volume[0],
                               rtol=CONSERVATION_RTOL)


def _inventory(result, order):
    """Crystal inventory of one moment over a solve.

    Parameters
    ----------
    result : DynamicResult
        Result of a solved crystallizer.
    order : int
        Moment order k.

    Returns
    -------
    numpy.ndarray
        mu_k times the slurry volume [um**k], one value per time.
    """
    return np.asarray(result.mu_n_solid0)[:, order] * np.ravel(result.vessel_vol)


def _flowsheet(upstream, downstream):
    """Solve two units connected in series.

    Parameters
    ----------
    upstream, downstream : MultiPhaseVessel
        Source and destination units.

    Returns
    -------
    SimulationExec
        The solved flowsheet, with units ``U01`` and ``U02``.
    """
    flowsheet = SimulationExec(DATA_PATH, flowsheet="U01 --> U02")
    flowsheet.U01, flowsheet.U02 = upstream, downstream
    flowsheet.SolveFlowsheet(
        kwargs_run={"U01": {"runtime": RUNTIME, "verbose": False},
                    "U02": {"runtime": RUNTIME, "verbose": False}},
        verbose=False)
    return flowsheet


def _washing_out_seeds(mechanism_cls=QMOMPopulationBalance):
    """Continuous crystallizer whose seeds only wash out.

    Parameters
    ----------
    mechanism_cls : type, optional
        Population balance class.

    Returns
    -------
    ContinuousCrystallizer
        Unit holding ``SEED_COUNT`` seeds per m**3, with no growth or
        nucleation, ready for a flowsheet.
    """
    vessel, _, _ = _build(SEED_COUNT * SEED_SIZE ** np.arange(NUM_MOM), None,
                          mechanism_cls=mechanism_cls,
                          vessel_cls=ContinuousCrystallizer, feed_mass_flow=SLOW_FEED,
                          growth_rate=0.0, nucleation_rate=0.0, integrator=_lsoda())
    return vessel


def _receiver(num_mom=NUM_MOM, mechanism_cls=QMOMPopulationBalance, **kernels):
    """Empty semibatch crystallizer with no growth or nucleation.

    Parameters
    ----------
    num_mom : int, optional
        Number of moments.
    mechanism_cls : type, optional
        Population balance class.
    **kernels
        Kernels for the population balance.

    Returns
    -------
    SemiBatchCrystallizer
        Unit ready for a flowsheet.
    """
    vessel, _, _ = _build(np.zeros(num_mom), None, mechanism_cls=mechanism_cls,
                          vessel_cls=SemiBatchCrystallizer, feed_mass_flow=SLOW_FEED,
                          growth_rate=0.0, nucleation_rate=0.0, integrator=_lsoda(), **kernels)
    return vessel


@pytest.mark.integration
@pytest.mark.parametrize("upstream_cls, downstream_cls", [
    (QMOMPopulationBalance, QMOMPopulationBalance),
    (MomentsPopulationBalance, QMOMPopulationBalance),
    (QMOMPopulationBalance, MomentsPopulationBalance),
], ids=["qmom_to_qmom", "moment_form_to_qmom", "qmom_to_moment_form"])
def test_crystals_leaving_one_moment_unit_arrive_in_the_next(upstream_cls, downstream_cls):
    flowsheet = _flowsheet(_washing_out_seeds(upstream_cls),
                           _receiver(mechanism_cls=downstream_cls))

    for order in (0, 3):
        upstream = _inventory(flowsheet.U01.result, order)
        lost = upstream[0] - upstream[-1]
        # Guard against a vacuous pass: a good part of the seeds did leave.
        assert lost > 0.25 * upstream[0]
        assert _inventory(flowsheet.U02.result, order)[-1] == pytest.approx(
            lost, rel=FLOW_SOLVE_RTOL)


@pytest.mark.integration
def test_a_feed_keeps_the_kernels_of_the_receiving_unit():
    # Seeds wash out at 1/tau and arrive to break at S_0, each event adding a
    # crystal: N(T) = (N_0 / tau) exp(S_0 T) (1 - exp(-(S_0 + 1/tau) T)) / (S_0 + 1/tau).
    # Breakage conserves the crystal volume, which still arrives in full.
    selection_rate = 1e-3  # [1/s]
    flowsheet = _flowsheet(
        _washing_out_seeds(),
        _receiver(breakage=ConstantSelection(selection_rate), fragments=BinaryFragmentation()))

    upstream = _inventory(flowsheet.U01.result, 0)
    tau = _residence_time(flowsheet.U01)
    decay = selection_rate + 1 / tau  # [1/s]
    expected = (upstream[0] / tau * np.exp(selection_rate * RUNTIME)
                * (1 - np.exp(-decay * RUNTIME)) / decay)
    assert _inventory(flowsheet.U02.result, 0)[-1] == pytest.approx(expected, rel=FLOW_SOLVE_RTOL)

    volume_upstream = _inventory(flowsheet.U01.result, 3)
    assert _inventory(flowsheet.U02.result, 3)[-1] == pytest.approx(
        volume_upstream[0] - volume_upstream[-1], rel=FLOW_SOLVE_RTOL)


@pytest.mark.integration
@pytest.mark.parametrize("mechanism_cls, num_mom", [
    (QMOMPopulationBalance, 4), (MomentsPopulationBalance, 4),
], ids=["fewer_nodes", "moment_form"])
def test_a_feed_with_another_moment_count_is_refused(mechanism_cls, num_mom):
    with pytest.raises(PharmaPyValueError, match="carrying 6 moments .* carries 4"):
        _flowsheet(_washing_out_seeds(), _receiver(num_mom, mechanism_cls))


@pytest.mark.integration
def test_a_feed_carrying_a_resolved_distribution_is_refused():
    # It would otherwise be skipped, and its crystals lost without a word.
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)
    grid = np.arange(1.0, 200.0)  # [um]
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components=TARGET, solvent_name="solvent", x_grid=grid,
        distrib_init=1e8 * np.exp(-0.5 * ((grid - SEED_SIZE) / 5.0) ** 2),  # [#/(um m**3)]
        density=LIQUID_SOLUTE_DENSITY, kv=SHAPE_FACTOR)
    upstream = ContinuousCrystallizer(integrator=_lsoda(), isothermal=True)
    upstream.Phases = [LiquidPhase(DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC,
                                   temp=TEMP, name_solv="solvent"), solid]
    upstream.CrystKinetics = _kinetics(None, growth_rate=0.0, nucleation_rate=0.0)
    upstream.Inlet = LiquidStream(DATA_PATH, mass_flow=SLOW_FEED, mass_frac=CHARGE_MASS_FRAC,
                                  temp=TEMP, name_solv="solvent")

    with pytest.raises(PharmaPySpecificationError, match="'distrib'"):
        _flowsheet(upstream, _receiver())


@pytest.mark.integration
def test_a_charge_of_crystals_into_a_qmom_vessel_is_refused():
    upstream, _, _ = _build(SEED_COUNT * SEED_SIZE ** np.arange(NUM_MOM), None,
                            growth_rate=0.0, nucleation_rate=0.0, integrator=_lsoda())
    downstream, _, _ = _build(np.zeros(NUM_MOM), None, growth_rate=0.0, nucleation_rate=0.0,
                              integrator=_lsoda())

    with pytest.raises(PharmaPySpecificationError, match="charge of\\s+crystals"):
        _flowsheet(upstream, downstream)


def _legacy_solid():
    """Pre-refactor solid carrying moments.

    Returns
    -------
    PharmaPy.Phases.SolidPhase
        Seeds in 1 L of slurry; total moments in metres [m**k], as that
        stack holds them.
    """
    slurry_volume = 1e-3  # [m**3]
    totals = (SEED_COUNT * slurry_volume
              * (SEED_SIZE * METRE_PER_MICRON) ** np.arange(4))  # [m**k]
    return OldSolidPhase(DATA_PATH, moments=totals, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)


def _legacy_solid_stream():
    """Pre-refactor solid stream carrying a distribution.

    Returns
    -------
    PharmaPy.Streams.SolidStream
        Crystals flowing around ``SEED_SIZE``, at number rates [#/um/s].
    """
    sizes = np.array([40.0, 50.0, 60.0])  # [um]
    return OldSolidStream(DATA_PATH, temp=TEMP, mass_frac=CRYSTAL_MASS_FRAC,
                          distrib=np.array([1e3, 2e3, 1e3]), x_distrib=sizes)


def _legacy_liquid():
    """Pre-refactor liquid of the charge's composition.

    Returns
    -------
    PharmaPy.Phases.LiquidPhase
        Liquid of mass ``CHARGE_MASS`` at ``TEMP``.
    """
    return OldLiquidPhase(DATA_PATH, temp=TEMP, mass=CHARGE_MASS,
                          mass_frac=CHARGE_MASS_FRAC, name_solv="solvent")


def _legacy_slurry():
    """Pre-refactor slurry of liquid and seeds.

    Returns
    -------
    PharmaPy.MixedPhases.Slurry
        Slurry holding ``_legacy_liquid`` and ``_legacy_solid``.
    """
    slurry = OldSlurry()
    slurry.Phases = (_legacy_liquid(), _legacy_solid())
    return slurry


@pytest.mark.parametrize("make_matter", [_legacy_solid, _legacy_solid_stream, _legacy_slurry],
                         ids=["solid", "solid_stream", "slurry"])
def test_legacy_crystals_are_refused_by_a_qmom_vessel(make_matter):
    vessel, _, _ = _build(np.zeros(NUM_MOM), None, agglomeration=ConstantKernel(1e-14))

    with pytest.raises(PharmaPySpecificationError, match="legacy unit"):
        Connection(None, vessel)._match_stacks(make_matter())


def test_a_legacy_liquid_charges_a_qmom_vessel_and_keeps_its_kernels():
    # Only agglomeration acts, on the configured crystals, so the count falls
    # as beta_0 mu_0**2 / 2 whichever liquid surrounds them.
    kernel_constant = 1e-14  # [m**3/s]
    moments = _population_moments(*THREE_SIZES)
    vessel, mechanism, _ = _agglomeration_only(moments, ConstantKernel(kernel_constant))

    liquid = Connection(None, vessel)._match_stacks(_legacy_liquid())
    liquid.transferred_from_uo = True
    vessel.Phases = liquid
    vessel.compile_structure()

    _, state, rates = _state_and_rates(vessel)

    assert vessel.Phases.Solids[0].get_mechanism(QMOMPopulationBalance) is mechanism
    assert rates[0] == pytest.approx(-kernel_constant * state[0] ** 2 / 2, rel=ALGEBRAIC_RTOL)


def test_qmom_hands_its_moments_to_the_legacy_stack_as_totals():
    moments = _population_moments(*THREE_SIZES)
    _, mechanism, liquid = _build(moments, None, shape_factor=SHAPE_FACTOR)

    handed = mechanism.to_legacy_state()

    phi = SHAPE_FACTOR * moments[3] * METRE_PER_MICRON ** 3  # [-]
    slurry_volume = float(liquid.vol) / (1 - phi)  # [m**3]
    expected = moments * slurry_volume * METRE_PER_MICRON ** np.arange(NUM_MOM)  # [m**k]
    assert handed["num_mom"] == NUM_MOM
    np.testing.assert_allclose(handed["moments"], expected, rtol=ALGEBRAIC_RTOL)


def _solved_without_kinetics(moments):
    """Solve a batch in which the crystals do nothing, to read its outputs.

    Parameters
    ----------
    moments : numpy.ndarray
        Moments [um**k/m**3 of slurry], held constant.

    Returns
    -------
    DynamicResult
        Result of a 1 s solve.
    """
    vessel, _, _ = _build(moments, None, growth_rate=0.0, nucleation_rate=0.0,
                          integrator=_lsoda())
    vessel.solve_unit(runtime=1.0, verbose=False)
    return vessel.result


def test_a_solve_reports_the_quadrature_and_the_mean_sizes():
    sizes, counts = THREE_SIZES
    result = _solved_without_kinetics(_population_moments(sizes, counts))

    order = np.argsort(sizes)
    nodes = np.asarray(result.quadrature_nodes)  # [um], one row per time
    weights = np.asarray(result.quadrature_weights)  # [#/m**3], one row per time
    np.testing.assert_allclose(nodes, np.tile(np.asarray(sizes)[order], (len(nodes), 1)),
                               rtol=ALGEBRAIC_RTOL)
    np.testing.assert_allclose(weights, np.tile(np.asarray(counts)[order], (len(weights), 1)),
                               rtol=ALGEBRAIC_RTOL)

    sizes, counts = np.asarray(sizes), np.asarray(counts)
    third, second = np.sum(counts * sizes ** 3), np.sum(counts * sizes ** 2)  # [um**k/m**3]
    np.testing.assert_allclose(np.ravel(result.L32), third / second, rtol=ALGEBRAIC_RTOL)
    np.testing.assert_allclose(np.ravel(result.L43), np.sum(counts * sizes ** 4) / third,
                               rtol=ALGEBRAIC_RTOL)


def test_outputs_pad_the_nodes_a_monodisperse_population_does_not_use():
    result = _solved_without_kinetics(SEED_COUNT * SEED_SIZE ** np.arange(NUM_MOM))

    np.testing.assert_allclose(np.asarray(result.quadrature_nodes)[-1],
                               [SEED_SIZE, np.nan, np.nan], rtol=ALGEBRAIC_RTOL)
    np.testing.assert_allclose(np.asarray(result.quadrature_weights)[-1],
                               [SEED_COUNT, np.nan, np.nan], rtol=ALGEBRAIC_RTOL)
    assert np.ravel(result.L43)[-1] == pytest.approx(SEED_SIZE, rel=ALGEBRAIC_RTOL)


def test_outputs_are_undefined_without_crystals():
    result = _solved_without_kinetics(np.zeros(NUM_MOM))

    assert np.all(np.isnan(result.quadrature_nodes))
    assert np.all(np.isnan(result.quadrature_weights))
    assert np.all(np.isnan(result.L32))
    assert np.all(np.isnan(result.L43))


def test_mean_sizes_are_undefined_when_solver_noise_leaves_no_denominator():
    # Round-off can leave a moment just below zero and the next just above: a
    # ratio of the two would report a negative size.
    noise = 1e-20  # [um**k/m**3]
    _, mechanism, _ = _build(np.zeros(NUM_MOM), None)
    key = StateKey(mechanism.moments_state_name, mechanism.owning_phase_ref)

    l32 = mechanism.compute_l32_output(
        None, 0.0, {key: np.array([0.0, 0.0, -noise, noise, noise, noise])}, None)
    l43 = mechanism.compute_l43_output(
        None, 0.0, {key: np.array([0.0, 0.0, 0.0, -noise, noise, noise])}, None)

    assert np.isnan(l32).all()
    assert np.isnan(l43).all()


def test_two_nodes_report_no_l43_because_mu_4_is_not_carried():
    moments = np.array([np.sum(np.asarray(FEW_SIZES[1]) * np.asarray(FEW_SIZES[0]) ** k)
                        for k in range(4)])  # [um**k/m**3]

    result = _solved_without_kinetics(moments)

    assert not hasattr(result, "L43")
    assert np.asarray(result.quadrature_nodes).shape[1] == 2
    assert np.ravel(result.L32)[-1] == pytest.approx(moments[3] / moments[2], rel=ALGEBRAIC_RTOL)
