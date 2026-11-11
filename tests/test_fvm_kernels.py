"""Agglomeration and breakage on the refactored finite-volume population balance.

The FVM batch crystallizer runs isothermal, with constant-rate kinetics, so
the references are closed form: collision and fragment counting on one
right-hand side, splits derived by hand on a grid whose node volumes
double, and over a solve Scott's (1968) exact solutions for the constant
and sum kernels and Ziff and McGrady's (1985) for breakage. The crystal
density equals the liquid density of the solute, as in
``test_qmom_population_balance``.

Units: mass [kg], sizes [um], number density [#/(um m**3 of slurry)], cell
counts [#/m**3 of slurry], rates per second. The solves use SciPy LSODA on
100 cells.
"""

import os
import warnings

import numpy as np
import pytest
from scipy.special import gamma

from PharmaPy.CrystallizationKernels import (BinaryFragmentation, BrownianKernel, ConstantKernel,
                                             ConstantSelection, DifferentialSedimentationKernel,
                                             ErosionFragmentation, KernelConditions,
                                             ParabolicFragmentation, PowerLawSelection,
                                             ProductKernel, ShearKernel, SumKernel,
                                             UniformFragmentation)
from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.DataClasses import StateKey
from PharmaPy.Errors import PharmaPySpecificationError, PharmaPyTypeError, PharmaPyValueError
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import OneDFVMMechanism, QMOMPopulationBalance
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
# The charge holds about 187 kg/m**3 of C, so it stays supersaturated.
SOLUBILITY = 50.0  # [kg/m**3]
LIQUID_SOLUTE_DENSITY = 1200.0  # [kg/m**3], the database's liquid density of C
GROWTH_RATE = 0.1  # G0 [um/s]
NUCLEATION_RATE = 1e8  # B [#/(m**3 s)]
DISSOLUTION_RATE = -0.05  # D [um/s], negative as CrystKinetics returns it
METRE_PER_MICRON = 1e-6  # [m/um], exact
# A sphere of diameter L has volume (pi/6) L**3; unequal to one, so a lost
# shape factor shows in the size-dependent kernels.
SPHERE_SHAPE_FACTOR = np.pi / 6  # [-]

# 100 geometric cells over 1-1000 um, the grid the cell average was measured on.
GRID = np.geomspace(1.0, 1000.0, 100)  # [um]
# Each node holds twice the volume of the one below (Hounslow, Ryall and
# Marshall, 1988, AIChE J. 34:1821-1832), so the split can be derived by hand.
DOUBLING_GRID = 10.0 * 2.0 ** (np.arange(8) / 3)  # [um]

# One right-hand side, compared through the rounding of sums over all pairs.
ALGEBRAIC_RTOL = 1e-10  # [-]
SOLVER_TOL = 1e-10  # [-], rtol of the LSODA solves
# atol of the LSODA solves, far below the densities that carry the crystals,
# 1e5 to 1e8. The mechanism's default, 1e4, left the constant kernel's mu_0
# 8.4e-5 from exact; 1e2 left 9.7e-7, 1 left 1.0e-8 (10 Nov 2026).
SOLVER_ATOL = 1e-2  # [#/(um m**3)]
# The cell average and the fixed pivot keep the count exact, so mu_0
# carries only integration error: 4.5e-11 for the constant kernel, 9.1e-11
# for the sum kernel and 1.2e-12 for symmetric breakage at SOLVER_ATOL
# (10 Nov 2026), a margin of 11.
EXACT_ZEROTH_MOMENT_RTOL = 1e-9  # [-]
# Agglomeration and breakage conserve the mechanism's mu_3 to rounding:
# 1.8e-15 at most alone, 2.0e-13 over the solve with both (10 Nov 2026).
MU3_CONSERVATION_RTOL = 1e-12  # [-]
# Volume-weighted error of the cell counts against Scott's solution at
# tau = 10: 5.40e-3 for the cell average on GRID, independent of the solver
# tolerances above, against 1.6e-2 for fixed pivot (Kumar and Ramkrishna,
# 1996) on the same grid (10 Nov 2026). The bound tells the two apart.
CELL_AVERAGE_DISTRIBUTION_TOL = 1e-2  # [-]
# The same error against Ziff and McGrady's breakage solution: 3.40e-3 for
# the fixed pivot on GRID at 1 + tau = 10, and 3.398e-3 at 50 (10 Nov 2026).
FIXED_PIVOT_BREAKAGE_TOL = 5e-3  # [-]

# Scott's (1968) constant-kernel case: an exponential start in volume L**3
# with mean MEAN_VOLUME; tau = beta_0 N_0 t reaches 10 over the solve.
MEAN_VOLUME = 50.0 ** 3  # [um**3]
INITIAL_COUNT = 1e10  # N_0 [#/m**3 of slurry]; kv * mu_3 * 1e-18 = 1.25e-3 [-]
RUNTIME = 600.0  # [s]
FINAL_TAU = 10.0  # [-]


def constant_rate(sup_sat, conc_sat, moments, temp, temp_ref, params):
    """Rate law independent of the state, for closed-form references.

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
        ``params[0]``: G0 [um/s], B [#/(m**3 s)] or D [um/s].
    """
    return params[0]


def _kinetics(growth_rate=0.0, nucleation_rate=0.0, growth_law=constant_rate):
    """Constant-rate nucleation and dissolution, and growth by ``growth_law``.

    Parameters
    ----------
    growth_rate : float, optional
        G0 [um/s], the growth law's first parameter. Defaults to zero.
    nucleation_rate : float, optional
        B [#/(m**3 s)]. Defaults to zero.
    growth_law : callable, optional
        Custom growth mechanism. Defaults to ``constant_rate``.

    Returns
    -------
    CrystKinetics
        Kinetics with growth and dissolution, so both transfer directions
        are wired.
    """
    return CrystKinetics(
        coeff_solub=[SOLUBILITY],
        nucl_prim=(nucleation_rate, 0.0, 0.0),
        growth=(growth_rate, 0.0, 1.0),
        dissolution=(DISSOLUTION_RATE, 0.0, 1.0),
        custom_mechanisms={"nucl_prim": constant_rate, "growth": growth_law,
                           "dissolution": constant_rate},
    )


def _faces(sizes):
    """Cell faces of a uniform or geometric grid [um].

    On a uniform grid every cell spans one spacing h centred on its node, so
    its faces lie h/2 either side. On a geometric grid every cell spans one
    grid ratio r centred geometrically on its node, so its faces are the node
    divided and multiplied by sqrt(r).

    Parameters
    ----------
    sizes : numpy.ndarray
        Uniform or geometric size grid [um], shape ``(n,)``.

    Returns
    -------
    numpy.ndarray
        Faces [um], shape ``(n + 1,)``.
    """
    spacing = sizes[1] - sizes[0]  # [um]
    if np.allclose(np.diff(sizes), spacing):
        return np.append(sizes, sizes[-1] + spacing) - spacing / 2

    ratio = sizes[1] / sizes[0]  # [-]
    return np.append(sizes, sizes[-1] * ratio) / np.sqrt(ratio)


def _build(counts, kernel, sizes=GRID, shape_factor=1.0, integrator=None, rad=None,
           breakage=None, fragments=None, **kinetics):
    """Build the closed FVM batch crystallizer from cell counts.

    Parameters
    ----------
    counts : numpy.ndarray
        Crystals in each cell [#/m**3 of slurry], shape ``sizes.shape``.
    kernel : callable or None
        Agglomeration kernel.
    sizes : numpy.ndarray, optional
        Uniform or geometric size grid [um]. Defaults to ``GRID``.
    shape_factor : float, optional
        Volume shape factor kv [-]. Defaults to one.
    integrator : IntegratorBackend, optional
        Backend for a solve. None for solver-free evaluation.
    rad : float, optional
        Nucleus size [um]. None, the default, keeps the first node.
    breakage, fragments : optional
        Selection function and fragment distribution; None for no breakage.
    **kinetics
        Passed to ``_kinetics``; growth and nucleation default to zero.

    Returns
    -------
    tuple
        The compiled crystallizer, its population balance and its liquid.
    """
    liquid = LiquidPhase(DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC,
                         temp=TEMP, name_solv="solvent")
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)
    mechanism = OneDFVMMechanism(
        owning_phase=solid, target_components=TARGET, solvent_name="solvent",
        x_grid=sizes, distrib_init=counts / np.diff(_faces(sizes)),
        density=LIQUID_SOLUTE_DENSITY, kv=shape_factor, rad=rad, agglomeration=kernel,
        breakage=breakage, fragments=fragments)
    solid.mechanisms = mechanism

    vessel = BatchCrystallizer(integrator=integrator, isothermal=True)
    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = _kinetics(**kinetics)
    vessel.compile_structure()

    return vessel, mechanism, liquid


def _rates(vessel, sizes=GRID):
    """Evaluate the material balance once at t = 0.

    Parameters
    ----------
    vessel : BatchCrystallizer
        Compiled vessel.
    sizes : numpy.ndarray, optional
        Its size grid [um]. Defaults to ``GRID``.

    Returns
    -------
    tuple of numpy.ndarray
        Liquid species rates [kg/s], and the cell-count rates
        [#/(m**3 s)], the density rates times the cell widths.
    """
    states = np.asarray(vessel.create_solver_init_states(), dtype=float)
    rates = np.asarray(vessel.unit_model(0.0, states, mat_bce=True), dtype=float).copy()

    collection = vessel.solver_state_collection
    species_key = next(k for k in collection.states if k.name == "mass_j")
    density_key = next(k for k in collection.states if k.name == "distrib")

    return (rates[collection.slices[species_key]],
            rates[collection.slices[density_key]] * np.diff(_faces(sizes)))


def _conditions(shape_factor=1.0):
    """Conditions for evaluating a built-in kernel, which reads only temp and kv.

    Parameters
    ----------
    shape_factor : float, optional
        Volume shape factor [-]. Defaults to one.

    Returns
    -------
    KernelConditions
        At ``TEMP``.
    """
    return KernelConditions(temp=TEMP, supersaturation=0.0, growth_rate=0.0,
                            shape_factor=shape_factor)


def _pair_rates(beta, counts):
    """Collision rates of the populated cells, every unordered pair once.

    Parameters
    ----------
    beta : float
        Constant kernel [m**3/s].
    counts : dict
        Crystals [#/m**3] by cell index.

    Returns
    -------
    dict
        Collisions [#/(m**3 s)] by pair of cell indices (i, j), i <= j.
    """
    return {(i, j): beta * counts[i] * counts[j] * (0.5 if i == j else 1.0)
            for i in counts for j in counts if i <= j}


def _assert_count_rates(count_rates, expected, collisions):
    """Compare cell-count rates through the rounding of the collision sums.

    Parameters
    ----------
    count_rates : numpy.ndarray
        Computed rates [#/(m**3 s)], one per cell.
    expected : dict
        Expected rates [#/(m**3 s)] by cell index; every other cell zero.
    collisions : dict
        Collision rates [#/(m**3 s)], whose total sets the scale.
    """
    expected_rates = np.zeros(len(count_rates))
    for cell, rate in expected.items():
        expected_rates[cell] = rate
    gross = 2 * sum(collisions.values())  # [#/(m**3 s)], crystals lost
    np.testing.assert_allclose(count_rates, expected_rates, rtol=0, atol=ALGEBRAIC_RTOL * gross)


def test_agglomerates_are_split_by_the_mean_volume_of_their_cell():
    # Cells 2, 3 and 4 of a grid whose node volumes double, v_k = 2**k v_0,
    # with faces at sqrt(2) v_k: cell 4 (node 4 v_2) collects 2-3 pairs at
    # 3 v_2, 3-3 pairs at 4 v_2 and 2-4 pairs at 5 v_2, from both sides of
    # its node, and moves their net volume excess v_2 (e_24 - e_23), here
    # positive, to node 5 as (e_24 - e_23) / 4 crystals. Cell 5 (node 8 v_2)
    # collects 3-4 pairs at 6 v_2 and 4-4 pairs at 8 v_2, and moves e_34 / 2
    # down to node 4; 2-2 pairs land on node 3. Splitting each pair alone
    # would instead send e_23 / 2 to node 3 and e_24 / 4 to node 5.
    beta = 1e-14  # [m**3/s], arbitrary
    populated = {2: 3e9, 3: 1e9, 4: 2e9}  # [#/m**3], so that e_24 > e_23
    counts = np.zeros(len(DOUBLING_GRID))
    counts[list(populated)] = list(populated.values())
    vessel, _, _ = _build(counts, ConstantKernel(beta), sizes=DOUBLING_GRID)

    _, count_rates = _rates(vessel, DOUBLING_GRID)

    e = _pair_rates(beta, populated)  # [#/(m**3 s)]
    to_node_5 = (e[2, 4] - e[2, 3]) / 4
    expected = {
        2: -2 * e[2, 2] - e[2, 3] - e[2, 4],
        3: e[2, 2] - e[2, 3] - 2 * e[3, 3] - e[3, 4],
        4: e[2, 3] + e[3, 3] + e[2, 4] - to_node_5 + e[3, 4] / 2
        - e[2, 4] - e[3, 4] - 2 * e[4, 4],
        5: e[3, 4] / 2 + e[4, 4] + to_node_5,
    }
    _assert_count_rates(count_rates, expected, e)


def test_agglomerates_are_split_by_the_mean_volume_of_their_cell_on_a_uniform_grid():
    # Nodes every 10 um, faces midway. Crystals at 20 and 30 um: cell 2
    # (node 27000 um**3, faces 15625 and 42875) collects 20-20 pairs at 16000
    # and 20-30 pairs at 35000, from both sides of its node, and moves their
    # net excess -11000 e_11 + 8000 e_12, here positive, to node 3 (64000)
    # over the spacing 37000. Cell 3 collects 30-30 pairs at 54000 and moves
    # 10000 e_22 / 37000 down to node 2.
    sizes = np.linspace(10.0, 80.0, 8)  # [um]
    beta = 1e-14  # [m**3/s], arbitrary
    populated = {1: 1e9, 2: 2e9}  # [#/m**3], so that the excess is positive
    counts = np.zeros(len(sizes))
    counts[list(populated)] = list(populated.values())
    vessel, _, _ = _build(counts, ConstantKernel(beta), sizes=sizes)

    _, count_rates = _rates(vessel, sizes)

    e = _pair_rates(beta, populated)  # [#/(m**3 s)]
    to_node_3 = (-11000 * e[1, 1] + 8000 * e[1, 2]) / 37000
    to_node_2 = 10000 * e[2, 2] / 37000
    expected = {
        1: -2 * e[1, 1] - e[1, 2],
        2: e[1, 1] + e[1, 2] - to_node_3 + to_node_2 - e[1, 2] - 2 * e[2, 2],
        3: e[2, 2] - to_node_2 + to_node_3,
    }
    _assert_count_rates(count_rates, expected, e)


def _spread_counts(sizes=GRID):
    """A log-normal population of cells, empty above 300 um.

    Parameters
    ----------
    sizes : numpy.ndarray, optional
        Size grid [um] reaching past 378 um. Defaults to ``GRID``.

    Returns
    -------
    numpy.ndarray
        Cell counts [#/m**3]. Two crystals of 300 um make one of 378 um, so
        no agglomerate passes the last node. A node at zero size is empty.
    """
    populated = (sizes > 0) & (sizes <= 300.0)
    counts = np.zeros(len(sizes))  # [#/m**3]
    counts[populated] = 1e10 * np.exp(-0.5 * (np.log(sizes[populated] / 40.0) / 0.6) ** 2)
    return counts


# A uniform grid, whose cells the split must handle as well.
UNIFORM_GRID = np.linspace(5.0, 500.0, 100)  # [um]


# Viscosity of the order of water's and a moderate density difference, as in
# test_qmom_population_balance; the kernel strengths are otherwise arbitrary.
VISCOSITY = 1e-3  # [Pa s]
DENSITY_DIFFERENCE = 300.0  # [kg/m**3]
AGGLOMERATION_KERNELS = {
    "constant": ConstantKernel(1e-14),
    "brownian": BrownianKernel(VISCOSITY),
    "sum": SumKernel(0.5),
    "shear": ShearKernel(20.0, 0.4),
    "sedimentation": DifferentialSedimentationKernel(DENSITY_DIFFERENCE, VISCOSITY, 0.01),
    "product": ProductKernel(1e14),
}


@pytest.mark.parametrize("sizes", [GRID, UNIFORM_GRID], ids=["geometric", "uniform"])
@pytest.mark.parametrize("name", AGGLOMERATION_KERNELS)
def test_each_collision_removes_one_crystal_and_keeps_the_volume(name, sizes):
    kernel = AGGLOMERATION_KERNELS[name]
    counts = _spread_counts(sizes)
    vessel, _, _ = _build(counts, kernel, sizes=sizes, shape_factor=SPHERE_SHAPE_FACTOR)

    _, count_rates = _rates(vessel, sizes)

    beta = kernel(sizes[:, None], sizes[None, :], _conditions(SPHERE_SHAPE_FACTOR))  # [m**3/s]
    collisions = counts @ beta @ counts / 2  # [#/(m**3 s)], every unordered pair once
    assert collisions > 0
    assert np.sum(count_rates) == pytest.approx(-collisions, rel=ALGEBRAIC_RTOL)
    gross_volume = np.sum(np.abs(count_rates) * sizes ** 3)  # [um**3/(m**3 s)]
    assert np.sum(count_rates * sizes ** 3) == pytest.approx(0.0, abs=ALGEBRAIC_RTOL * gross_volume)


def test_agglomerates_past_the_last_node_keep_their_volume_in_the_last_cell():
    # The last two cells only: on GRID every such agglomerate lies above the
    # last face, so each collision removes its pair and leaves its volume
    # at the last node, as (v_i + v_j) / v_last crystals.
    beta = 1e-14  # [m**3/s], arbitrary
    counts = np.zeros(len(GRID))
    counts[-2:] = 2e8, 5e7  # [#/m**3]
    volumes = GRID[-2:] ** 3  # [um**3]
    vessel, _, _ = _build(counts, ConstantKernel(beta))

    _, count_rates = _rates(vessel)

    pairs = [(beta * counts[-2] ** 2 / 2, (0, 0)), (beta * counts[-2] * counts[-1], (0, 1)),
             (beta * counts[-1] ** 2 / 2, (1, 1))]  # [#/(m**3 s)], cells counted from -2
    expected = np.zeros(2)
    for frequency, (first, second) in pairs:
        expected[first] -= frequency
        expected[second] -= frequency
        expected[1] += frequency * (volumes[first] + volumes[second]) / volumes[1]
    gross = 2 * sum(frequency for frequency, _ in pairs)  # [#/(m**3 s)], crystals lost
    np.testing.assert_allclose(count_rates[-2:], expected, rtol=0, atol=ALGEBRAIC_RTOL * gross)
    np.testing.assert_array_equal(count_rates[:-2], 0.0)


def test_agglomeration_is_counted_once_and_moves_no_solute():
    # Growth, nucleation and both transfer directions run as well; the
    # kernel must add exactly its own term to the distribution, once, and
    # leave the liquid alone.
    counts = _spread_counts()
    kinetics = {"growth_rate": GROWTH_RATE, "nucleation_rate": NUCLEATION_RATE}
    with_kernel, _, _ = _build(counts, ConstantKernel(1e-14), **kinetics)
    without_kernel, _, _ = _build(counts, None, **kinetics)

    species_with, counts_with = _rates(with_kernel)
    species_without, counts_without = _rates(without_kernel)

    np.testing.assert_array_equal(species_with, species_without)
    collisions = 1e-14 * np.sum(counts) ** 2 / 2  # [#/(m**3 s)]
    assert np.sum(counts_with - counts_without) == pytest.approx(-collisions, rel=ALGEBRAIC_RTOL)


# A reference first moment [m/m**3] for the moment-paced growth law, of the
# order of _spread_counts' own, so the growth rate stays of the order of G0.
FIRST_MOMENT_REFERENCE = 1e6  # [m/m**3]


def moment_paced_rate(sup_sat, conc_sat, moments, temp, temp_ref, params):
    """Growth proportional to the crystals' first moment, so the moment basis shows.

    Parameters
    ----------
    sup_sat, conc_sat, temp, temp_ref
        Ignored.
    moments : numpy.ndarray
        Moments [m**k/m**3] the population balance hands the kinetics.
    params : array-like
        The growth parameters; the first is G0 [um/s].

    Returns
    -------
    float
        G0 mu_1 / ``FIRST_MOMENT_REFERENCE`` [um/s].
    """
    return params[0] * moments[1] / FIRST_MOMENT_REFERENCE


class ConditionRecordingKernel:
    """Constant agglomeration that keeps the conditions and smallest size it was given."""

    def __init__(self):
        self.seen = []
        self.smallest = np.inf  # [um]

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um], the smallest recorded.
        conditions : KernelConditions
            Recorded.

        Returns
        -------
        numpy.ndarray
            1e-14 [m**3/s], an arbitrary constant, for every pair.
        """
        self.seen.append(conditions)
        self.smallest = min(self.smallest, float(np.min(size_i)), float(np.min(size_j)))
        return np.full(np.broadcast(size_i, size_j).shape, 1e-14)


class SizeRecordingSelection:
    """Constant breakage frequency that keeps the smallest size it was asked about."""

    def __init__(self):
        self.smallest = np.inf  # [um]

    def __call__(self, size, conditions):
        """Breakage frequency.

        Parameters
        ----------
        size : numpy.ndarray
            Crystal sizes [um], the smallest recorded.
        conditions : KernelConditions
            Unused.

        Returns
        -------
        numpy.ndarray
            1e-3 [1/s], an arbitrary constant, for every size.
        """
        self.smallest = min(self.smallest, float(np.min(size)))
        return np.full(np.shape(size), 1e-3)


@pytest.mark.parametrize("kind", ["agglomeration", "breakage"])
def test_kernels_see_no_crystal_smaller_than_a_nucleus(kind):
    # As in QMOM: a nucleus size above the first node is the smallest size
    # the kernel or selection function is asked about.
    nucleus_size = 2.0 * GRID[0]  # [um]
    if kind == "agglomeration":
        recorder = ConditionRecordingKernel()
        kernels = {"kernel": recorder}
    else:
        recorder = SizeRecordingSelection()
        kernels = {"kernel": None, "breakage": recorder, "fragments": BinaryFragmentation()}
    vessel, _, _ = _build(_spread_counts(), rad=nucleus_size, **kernels)

    _rates(vessel)

    assert recorder.smallest == nucleus_size


def test_a_user_kernel_sees_the_state_it_acts_on():
    kernel = ConditionRecordingKernel()
    counts = _spread_counts()
    vessel, _, liquid = _build(counts, kernel, shape_factor=SPHERE_SHAPE_FACTOR,
                               growth_rate=GROWTH_RATE, growth_law=moment_paced_rate)

    _rates(vessel)

    # Relative supersaturation on the default basis, kg of C per m**3 of
    # solution, against the constant solubility; mu_1 in metres.
    concentration = CHARGE_MASS * CHARGE_MASS_FRAC[TARGET_INDEX] / float(liquid.vol)  # [kg/m**3]
    first_moment = np.sum(counts * GRID) * METRE_PER_MICRON  # [m/m**3]
    conditions = kernel.seen[-1]
    assert conditions.temp == pytest.approx(TEMP, rel=ALGEBRAIC_RTOL)
    assert conditions.supersaturation == pytest.approx(
        (concentration - SOLUBILITY) / SOLUBILITY, rel=ALGEBRAIC_RTOL)
    assert conditions.growth_rate == pytest.approx(
        GROWTH_RATE * first_moment / FIRST_MOMENT_REFERENCE, rel=ALGEBRAIC_RTOL)
    assert conditions.shape_factor == SPHERE_SHAPE_FACTOR


def _scott_counts(tau):
    """Scott's (1968) constant-kernel solution integrated over each cell of GRID.

    n(v, t) = (N_0 / v_0) 4 / (2 + tau)**2 exp(-2 v / (v_0 (2 + tau))) in the
    volume v = L**3, from an exponential start of mean v_0.

    Parameters
    ----------
    tau : float
        beta_0 N_0 t [-].

    Returns
    -------
    numpy.ndarray
        Cell counts [#/m**3], shape ``GRID.shape``.
    """
    mean_volume = MEAN_VOLUME * (2 + tau) / 2  # [um**3]
    face_volumes = _faces(GRID) ** 3  # [um**3]
    total = 2 * INITIAL_COUNT / (2 + tau)  # [#/m**3]
    return total * -np.diff(np.exp(-face_volumes / mean_volume))


def _solved_counts(kernel, counts, **kernels):
    """Solve the batch, with no growth or nucleation, over RUNTIME.

    Parameters
    ----------
    kernel : callable or None
        Agglomeration kernel.
    counts : numpy.ndarray
        Initial cell counts [#/m**3].
    **kernels
        ``breakage`` and ``fragments``, passed to ``_build``.

    Returns
    -------
    numpy.ndarray
        Cell counts [#/m**3] over time, shape ``(times, len(GRID))``.
    """
    vessel, _, _ = _build(counts, kernel, integrator=ScipyBackend(
        method="LSODA", options={"rtol": SOLVER_TOL, "atol": SOLVER_ATOL}), **kernels)
    vessel.solve_unit(runtime=RUNTIME, verbose=False)
    return np.asarray(vessel.result.distrib_solid0) * np.diff(_faces(GRID))


@pytest.mark.integration
def test_constant_kernel_follows_scotts_distribution_over_a_solve():
    beta = FINAL_TAU / (INITIAL_COUNT * RUNTIME)  # [m**3/s]
    start = _scott_counts(0.0)

    trajectory = _solved_counts(ConstantKernel(beta), start)

    final, exact = trajectory[-1], _scott_counts(FINAL_TAU)
    volumes = GRID ** 3  # [um**3]
    assert np.sum(final) == pytest.approx(np.sum(exact), rel=EXACT_ZEROTH_MOMENT_RTOL)
    np.testing.assert_allclose(trajectory @ volumes, start @ volumes, rtol=MU3_CONSERVATION_RTOL)
    volume_error = np.sum(np.abs(final - exact) * volumes) / np.sum(exact * volumes)  # [-]
    assert volume_error < CELL_AVERAGE_DISTRIBUTION_TOL


@pytest.mark.integration
def test_sum_kernel_follows_the_exact_zeroth_moment_over_a_solve():
    # dN/dt = -beta_0 phi N with phi = kv mu_3, so N falls by e over the solve.
    start = _scott_counts(0.0)
    phi = np.sum(start * GRID ** 3) * METRE_PER_MICRON ** 3  # [-], kv = 1
    rate_constant = 1 / (phi * RUNTIME)  # [1/s]

    trajectory = _solved_counts(SumKernel(rate_constant), start)

    assert np.sum(trajectory[-1]) == pytest.approx(
        np.sum(start) * np.exp(-1.0), rel=EXACT_ZEROTH_MOMENT_RTOL)
    np.testing.assert_allclose(trajectory @ GRID ** 3, start @ GRID ** 3,
                               rtol=MU3_CONSERVATION_RTOL)


@pytest.mark.parametrize("kernels", [
    {"kernel": ConstantKernel(1e-14)},
    {"kernel": None, "breakage": ConstantSelection(1e-3), "fragments": BinaryFragmentation()},
], ids=["agglomeration", "breakage"])
def test_kernels_left_out_of_the_vessel_are_refused(kernels):
    vessel, _, _ = _build(_spread_counts(), **kernels)

    vessel.intraphase_processes = []

    with pytest.raises(PharmaPySpecificationError, match="This OneDFVMMechanism has"):
        _rates(vessel)


def test_brownian_agglomeration_needs_a_first_node_above_zero_size():
    # Nuclei enter at the first node, where the Brownian kernel is infinite.
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    with pytest.raises(PharmaPyValueError, match="rad > 0"):
        OneDFVMMechanism(owning_phase=solid, target_components=TARGET, solvent_name="solvent",
                         x_grid=np.linspace(0.0, 100.0, 11), distrib_init=np.zeros(11),
                         agglomeration=BrownianKernel(VISCOSITY))


def test_an_agglomeration_kernel_must_be_callable():
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    with pytest.raises(PharmaPyTypeError, match="kernel callable"):
        OneDFVMMechanism(owning_phase=solid, target_components=TARGET, solvent_name="solvent",
                         x_grid=GRID, distrib_init=np.zeros(len(GRID)), agglomeration=1e-14)


def test_kernel_values_that_are_not_rates_are_refused():
    def negative(size_i, size_j, conditions):
        """A kernel returning a negative rate [m**3/s] for every pair."""
        return np.full(np.broadcast(size_i, size_j).shape, -1e-14)

    vessel, _, _ = _build(_spread_counts(), negative)

    with pytest.raises(PharmaPyValueError, match="agglomeration kernel must"):
        _rates(vessel)


BREAKAGE_RATE = 1e-3  # [1/s], arbitrary
# A grid from zero size, below whose first node no fragment can fall.
GRID_FROM_ZERO = np.linspace(0.0, 500.0, 101)  # [um]
FRAGMENT_DISTRIBUTIONS = {
    "symmetric": BinaryFragmentation(0.5),
    "mass_ratio_1_4": BinaryFragmentation(0.2),
    "erosion": ErosionFragmentation(7.3),  # [um], between nodes of the grids
    "parabolic": ParabolicFragmentation(1.0),
    "uniform": UniformFragmentation(),
}


def _size_paced_selection():
    """Breakage frequency proportional to size, so every cell breaks at its own rate.

    Returns
    -------
    PowerLawSelection
        a(L) = ``BREAKAGE_RATE`` L / (100 um) [1/s].
    """
    return PowerLawSelection(BREAKAGE_RATE, 1.0, reference_size=100.0)


def test_halves_of_a_parent_land_on_the_node_below_it():
    # On a grid whose node volumes double, each half of a parent at node k
    # is a crystal of node k - 1.
    counts = np.zeros(len(DOUBLING_GRID))
    counts[2], counts[5] = 3e9, 1e9  # [#/m**3]
    vessel, _, _ = _build(counts, None, sizes=DOUBLING_GRID, breakage=_size_paced_selection(),
                          fragments=BinaryFragmentation(0.5))

    _, count_rates = _rates(vessel, DOUBLING_GRID)

    broken = BREAKAGE_RATE * DOUBLING_GRID / 100.0 * counts  # [#/(m**3 s)]
    expected = -broken
    expected[1] += 2 * broken[2]
    expected[4] += 2 * broken[5]
    np.testing.assert_allclose(count_rates, expected, rtol=0,
                               atol=ALGEBRAIC_RTOL * 3 * np.sum(broken))


@pytest.mark.parametrize("sizes", [GRID, GRID_FROM_ZERO], ids=["geometric", "from_zero"])
@pytest.mark.parametrize("name", FRAGMENT_DISTRIBUTIONS)
def test_breakage_keeps_the_crystal_volume(name, sizes):
    vessel, _, _ = _build(_spread_counts(sizes), None, sizes=sizes,
                          breakage=_size_paced_selection(),
                          fragments=FRAGMENT_DISTRIBUTIONS[name])

    _, count_rates = _rates(vessel, sizes)

    gross_volume = np.sum(np.abs(count_rates) * sizes ** 3)  # [um**3/(m**3 s)]
    assert gross_volume > 0
    assert np.sum(count_rates * sizes ** 3) == pytest.approx(0.0, abs=ALGEBRAIC_RTOL * gross_volume)


@pytest.mark.parametrize("name", FRAGMENT_DISTRIBUTIONS)
def test_each_break_adds_its_fragments_less_the_parent(name):
    # No fragment falls below a first node at zero size, so each break
    # replaces its parent by b_0 crystals.
    fragments = FRAGMENT_DISTRIBUTIONS[name]
    counts = _spread_counts(GRID_FROM_ZERO)
    vessel, _, _ = _build(counts, None, sizes=GRID_FROM_ZERO, breakage=_size_paced_selection(),
                          fragments=fragments)

    _, count_rates = _rates(vessel, GRID_FROM_ZERO)

    broken = BREAKAGE_RATE * GRID_FROM_ZERO / 100.0 * counts  # [#/(m**3 s)]
    fragment_counts = fragments.moments(GRID_FROM_ZERO, [0])[:, 0]  # b_0 [-]
    expected = np.sum(broken * (fragment_counts - 1))  # [#/(m**3 s)]
    assert expected > 0
    assert np.sum(count_rates) == pytest.approx(expected, rel=ALGEBRAIC_RTOL)


def test_crystals_up_to_the_minimum_size_do_not_break():
    minimum_size = 60.0  # [um]
    counts = np.where(GRID <= minimum_size, _spread_counts(), 0.0)  # [#/m**3]
    vessel, _, _ = _build(counts, None, breakage=ConstantSelection(BREAKAGE_RATE, minimum_size),
                          fragments=BinaryFragmentation())
    breaking, _, _ = _build(counts, None, breakage=ConstantSelection(BREAKAGE_RATE),
                            fragments=BinaryFragmentation())

    _, count_rates = _rates(vessel)

    np.testing.assert_array_equal(count_rates, 0.0)
    assert np.any(_rates(breaking)[1] != 0)


@pytest.mark.integration
def test_constant_symmetric_breakage_doubles_the_count_over_a_solve():
    # dN/dt = a N while no fragment falls below the first node: with
    # a t = ln 2 the count doubles. Crystals from 30 um need fifteen halvings
    # to pass the first node, against 0.7 breaks per crystal over the solve;
    # from 10 um, ten halvings left the count 4e-9 short (10 Nov 2026).
    start = np.where(GRID >= 30.0, _spread_counts(), 0.0)  # [#/m**3]

    trajectory = _solved_counts(None, start, breakage=ConstantSelection(np.log(2) / RUNTIME),
                                fragments=BinaryFragmentation())

    assert np.sum(trajectory[-1]) == pytest.approx(2 * np.sum(start), rel=EXACT_ZEROTH_MOMENT_RTOL)
    np.testing.assert_allclose(trajectory @ GRID ** 3, start @ GRID ** 3,
                               rtol=MU3_CONSERVATION_RTOL)


def _ziff_mcgrady_counts(tau):
    """Ziff and McGrady's (1985) breakage solution integrated over each cell of GRID.

    With uniform binary fragments and a = S v, from an exponential start of
    mean v_0 in the volume v = L**3, n(v, t) = (N_0 / v_0) (1 + tau)**2
    exp(-(1 + tau) v / v_0), tau = S v_0 t; it satisfies the breakage
    equation by substitution.

    Parameters
    ----------
    tau : float
        S v_0 t [-].

    Returns
    -------
    numpy.ndarray
        Cell counts [#/m**3], shape ``GRID.shape``.
    """
    face_volumes = _faces(GRID) ** 3  # [um**3]
    total = INITIAL_COUNT * (1 + tau)  # [#/m**3]
    return total * -np.diff(np.exp(-(1 + tau) * face_volumes / MEAN_VOLUME))


@pytest.mark.integration
def test_volume_proportional_uniform_breakage_follows_ziff_and_mcgradys_distribution():
    # a = k (L / L_0)**3 = k v / v_0 with L_0**3 = v_0, so tau = k t; the
    # count grows tenfold over the solve.
    final_tau = 9.0  # [-]
    selection = PowerLawSelection(final_tau / RUNTIME, 3.0, reference_size=np.cbrt(MEAN_VOLUME))
    start = _ziff_mcgrady_counts(0.0)

    trajectory = _solved_counts(None, start, breakage=selection, fragments=UniformFragmentation())

    final, exact = trajectory[-1], _ziff_mcgrady_counts(final_tau)
    volumes = GRID ** 3  # [um**3]
    np.testing.assert_allclose(trajectory @ volumes, start @ volumes, rtol=MU3_CONSERVATION_RTOL)
    volume_error = np.sum(np.abs(final - exact) * volumes) / np.sum(exact * volumes)  # [-]
    assert volume_error < FIXED_PIVOT_BREAKAGE_TOL


class GridlessFragments:
    """Fragments with moments for QMOM but no counts on a grid."""

    def moments(self, parent_sizes, orders):
        """Moments of two halves of each parent.

        Parameters
        ----------
        parent_sizes : numpy.ndarray
            Parent sizes [um], shape ``(n,)``.
        orders : numpy.ndarray
            Moment orders, shape ``(m,)``.

        Returns
        -------
        numpy.ndarray
            b_k [um**k], shape ``(n, m)``.
        """
        return BinaryFragmentation().moments(parent_sizes, orders)


class ScaledGridFragments:
    """Binary halves whose grid counts are scaled, so they break the balance.

    Parameters
    ----------
    scale : float
        Factor on ``BinaryFragmentation().grid_counts`` [-].
    """

    def __init__(self, scale):
        self.scale = scale

    def grid_counts(self, grid_sizes):
        """Scaled counts of two halves of each parent.

        Parameters
        ----------
        grid_sizes : numpy.ndarray
            Node sizes [um], shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            Counts [-], shape ``(n, n)``, times ``scale``.
        """
        return self.scale * BinaryFragmentation().grid_counts(grid_sizes)


class TruncatedGridFragments:
    """Binary halves whose grid counts leave out the last node."""

    def grid_counts(self, grid_sizes):
        """Counts of two halves of each parent, without the last node's row.

        Parameters
        ----------
        grid_sizes : numpy.ndarray
            Node sizes [um], shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            Counts [-], shape ``(n - 1, n)``.
        """
        return BinaryFragmentation().grid_counts(grid_sizes)[:-1]


@pytest.mark.parametrize("kernels, error, message", [
    ({"breakage": ConstantSelection(BREAKAGE_RATE)}, PharmaPyValueError, "both a selection"),
    ({"breakage": ConstantSelection(BREAKAGE_RATE), "fragments": GridlessFragments()},
     PharmaPyTypeError, r"grid_counts\(grid_sizes\)"),
    ({"breakage": ConstantSelection(BREAKAGE_RATE), "fragments": ScaledGridFragments(0.5)},
     PharmaPyValueError, "hold their parent's volume"),
    ({"breakage": ConstantSelection(BREAKAGE_RATE), "fragments": ScaledGridFragments(-1.0)},
     PharmaPyValueError, "non-negative grid counts"),
    ({"breakage": ConstantSelection(BREAKAGE_RATE), "fragments": TruncatedGridFragments()},
     PharmaPyValueError, "grid counts of shape"),
], ids=["selection_alone", "fragments_without_grid_counts", "volume_losing", "negative",
        "wrong_shape"])
def test_incomplete_or_invalid_breakage_is_refused_at_construction(kernels, error, message):
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)

    with pytest.raises(error, match=message):
        OneDFVMMechanism(owning_phase=solid, target_components=TARGET, solvent_name="solvent",
                         x_grid=GRID, distrib_init=np.zeros(len(GRID)), **kernels)


def test_selection_values_that_are_not_rates_are_refused():
    def negative(size, conditions):
        """A selection function returning a negative frequency [1/s] for every size."""
        return np.full(np.shape(size), -BREAKAGE_RATE)

    vessel, _, _ = _build(_spread_counts(), None, breakage=negative,
                          fragments=BinaryFragmentation())

    with pytest.raises(PharmaPyValueError, match="selection function must return finite"):
        _rates(vessel)


@pytest.mark.integration
@pytest.mark.parametrize("top, crowded", [(150.0, True), (1000.0, False)],
                         ids=["short_grid", "wide_grid"])
def test_agglomerates_crowding_the_last_size_class_are_reported(top, crowded):
    # Crystals up to 60 um agglomerate to tau = 10 on 60 geometric cells.
    # Ending at 150 um, the last cell holds 0.187 % of the crystal volume by
    # the end; ending at 1000 um, none (10 Nov 2026).
    sizes = np.geomspace(1.0, top, 60)  # [um]
    counts = np.where(sizes <= 60.0, _spread_counts(sizes), 0.0)  # [#/m**3]
    beta = FINAL_TAU / (np.sum(counts) * RUNTIME)  # [m**3/s]
    vessel, _, _ = _build(counts, ConstantKernel(beta), sizes=sizes, integrator=ScipyBackend(
        method="LSODA", options={"rtol": SOLVER_TOL, "atol": SOLVER_ATOL}))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vessel.solve_unit(runtime=RUNTIME, verbose=False)

    reports = [str(warning.message) for warning in caught
               if "last size class held" in str(warning.message)]
    assert len(reports) == (1 if crowded else 0)
    if crowded:
        assert "%.6g um" % sizes[-1] in reports[0]


def test_the_largest_share_over_the_solve_is_reported():
    # A solved history whose last cell holds 0.5 % of the crystal volume
    # midway and none at either end, as breakage could leave it.
    vessel, mechanism, _ = _build(_spread_counts(), ConstantKernel(1e-14))
    process = next(process for process in vessel.intraphase_processes
                   if process.mechanism is mechanism.kernel_sources)
    widths = np.diff(_faces(GRID))  # [um]
    density = _spread_counts() / widths  # [#/(um m**3)]
    peaked = density.copy()
    rest_volume = np.sum(density * widths * GRID ** 3)  # [um**3/m**3]
    peaked[-1] = rest_volume * (0.005 / 0.995) / (widths[-1] * GRID[-1] ** 3)
    history = {StateKey("distrib", process.phaseref): np.array([density, peaked, density]),
               "time": np.array([0.0, 300.0, 600.0])}  # [s]

    with pytest.warns(RuntimeWarning, match="last size class held") as caught:
        mechanism.kernel_sources.review_solution(process, history)

    assert len(caught) == 1
    assert "0.5 %" in str(caught[0].message) and "t = 300 s" in str(caught[0].message)


# Patil and Andrews (1998) and Lage (2002): with a constant kernel beta_0,
# a = S v and uniform fragments, an exponential start stays exponential,
# n(v, t) = (N**2 / phi) exp(-N v / phi) in v = L**3 with phi = N_0 v_0 the
# crystal volume, because each term alone keeps that shape (Scott, 1968;
# Ziff and McGrady, 1985). The count then follows the Riccati equation
# dN/dt = S phi - beta_0 N**2 / 2, settling at N_inf = sqrt(2 S phi / beta_0)
# at the rate gamma = beta_0 N_inf / 2. Derived by substitution.
EQUILIBRIUM_COUNT = 2 * INITIAL_COUNT  # N_inf [#/m**3]
SETTLING_RATE = 3.0 / RUNTIME  # gamma [1/s]; tanh(3) = 0.995 by the end
# Errors of this case at RUNTIME on GRID (10 Nov 2026), each bound about
# twice its measurement: the FVM count 9.1e-4 from N(t) and its cells 2.43e-3
# from the exact distribution (volume-weighted); QMOM's moments up to 1.6e-3
# (mu_0 exact to 1e-12); the FVM's moments up to 5.3e-3, most of it already
# at the start, where the grid holds mu_5 of the exponential 5.1e-3 high.
BALANCE_COUNT_RTOL = 2e-3  # [-]
BALANCE_DISTRIBUTION_TOL = 5e-3  # [-]
QMOM_BALANCE_MOMENT_RTOL = 3e-3  # [-]
FVM_BALANCE_MOMENT_RTOL = 1e-2  # [-]


def _balance_kernels():
    """Kernels whose balance holds EQUILIBRIUM_COUNT, reached at SETTLING_RATE.

    Returns
    -------
    dict
        ``kernel``, ``breakage`` and ``fragments`` for ``_build``.
    """
    crystal_volume = INITIAL_COUNT * MEAN_VOLUME  # phi [um**3/m**3]
    beta = 2 * SETTLING_RATE / EQUILIBRIUM_COUNT  # beta_0 [m**3/s]
    selection_per_volume = beta * EQUILIBRIUM_COUNT ** 2 / (2 * crystal_volume)  # S [1/(s um**3)]
    return {"kernel": ConstantKernel(beta),
            "breakage": PowerLawSelection(selection_per_volume * MEAN_VOLUME, 3.0,
                                          reference_size=np.cbrt(MEAN_VOLUME)),
            "fragments": UniformFragmentation()}


def _balance_count(time):
    """Crystal count N(t) of the Riccati equation, from INITIAL_COUNT.

    Parameters
    ----------
    time : float
        Time [s].

    Returns
    -------
    float
        N [#/m**3].
    """
    settled = np.tanh(SETTLING_RATE * time)  # [-]
    return (EQUILIBRIUM_COUNT * (INITIAL_COUNT + EQUILIBRIUM_COUNT * settled)
            / (EQUILIBRIUM_COUNT + INITIAL_COUNT * settled))


def _balance_counts(time):
    """The exact distribution integrated over each cell of GRID.

    Parameters
    ----------
    time : float
        Time [s].

    Returns
    -------
    numpy.ndarray
        Cell counts [#/m**3], shape ``GRID.shape``.
    """
    count = _balance_count(time)  # [#/m**3]
    mean_volume = INITIAL_COUNT * MEAN_VOLUME / count  # phi / N [um**3]
    return count * -np.diff(np.exp(-_faces(GRID) ** 3 / mean_volume))


def _balance_moments(time, orders):
    """Moments of the exact distribution in size.

    Parameters
    ----------
    time : float
        Time [s].
    orders : numpy.ndarray
        Orders k [-].

    Returns
    -------
    numpy.ndarray
        mu_k = N (phi / N)**(k/3) Gamma(1 + k/3) [um**k/m**3].
    """
    count = _balance_count(time)  # [#/m**3]
    mean_volume = INITIAL_COUNT * MEAN_VOLUME / count  # [um**3]
    orders = np.asarray(orders, dtype=float)
    return count * mean_volume ** (orders / 3) * gamma(1 + orders / 3)


@pytest.mark.integration
def test_agglomeration_and_breakage_settle_on_their_exact_balance():
    start = _balance_counts(0.0)

    trajectory = _solved_counts(**_balance_kernels(), counts=start)

    final, exact = trajectory[-1], _balance_counts(RUNTIME)
    volumes = GRID ** 3  # [um**3]
    np.testing.assert_allclose(trajectory @ volumes, start @ volumes, rtol=MU3_CONSERVATION_RTOL)
    assert np.sum(final) == pytest.approx(_balance_count(RUNTIME), rel=BALANCE_COUNT_RTOL)
    volume_error = np.sum(np.abs(final - exact) * volumes) / np.sum(exact * volumes)  # [-]
    assert volume_error < BALANCE_DISTRIBUTION_TOL


QMOM_ORDERS = np.arange(6)  # [-], mu_0..mu_5, three quadrature nodes


def _qmom_moments_over_a_solve(kernels):
    """Solve the same batch with QMOMPopulationBalance.

    Parameters
    ----------
    kernels : dict
        ``kernel``, ``breakage`` and ``fragments``.

    Returns
    -------
    numpy.ndarray
        Moments [um**k/m**3] at the end of RUNTIME, shape ``(6,)``.
    """
    liquid = LiquidPhase(DATA_PATH, mass=CHARGE_MASS, mass_frac=CHARGE_MASS_FRAC,
                         temp=TEMP, name_solv="solvent")
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)
    solid.mechanisms = QMOMPopulationBalance(
        owning_phase=solid, target_components=TARGET, solvent_name="solvent",
        moments_init=_balance_moments(0.0, QMOM_ORDERS), density=LIQUID_SOLUTE_DENSITY,
        agglomeration=kernels["kernel"], breakage=kernels["breakage"],
        fragments=kernels["fragments"])
    vessel = BatchCrystallizer(integrator=ScipyBackend(
        method="LSODA", options={"rtol": SOLVER_TOL, "atol": SOLVER_TOL}), isothermal=True)
    vessel.Phases = [liquid, solid]
    vessel.CrystKinetics = _kinetics()
    vessel.solve_unit(runtime=RUNTIME, verbose=False)
    return np.asarray(vessel.result.mu_n_solid0)[-1]


@pytest.mark.integration
def test_fvm_and_qmom_agree_on_the_exact_balance():
    # The same case through both population balances: each moment against
    # the exact one. QMOM closes mu_0 exactly here; its other moments and
    # all the FVM's carry their method's error.
    kernels = _balance_kernels()
    exact = _balance_moments(RUNTIME, QMOM_ORDERS)

    qmom = _qmom_moments_over_a_solve(kernels)
    final_counts = _solved_counts(**kernels, counts=_balance_counts(0.0))[-1]
    fvm = np.array([np.sum(final_counts * GRID ** k) for k in QMOM_ORDERS])  # [um**k/m**3]

    np.testing.assert_allclose(qmom, exact, rtol=QMOM_BALANCE_MOMENT_RTOL)
    np.testing.assert_allclose(fvm, exact, rtol=FVM_BALANCE_MOMENT_RTOL)
