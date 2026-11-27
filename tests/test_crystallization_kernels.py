"""Built-in agglomeration and breakage kernels of ``PharmaPy.CrystallizationKernels``.

Expected rates are built from the crystal sizes converted to metres first,
the shear and Brownian kernels from Smoluchowski's radius forms and the
settling kernel from Stokes velocities, so none repeats the kernels' own
expressions; the degrees of homogeneity are those of Table 1 of Marchisio,
Vigil and Fox (2003). Fragment moments come from the fragments' sizes, or
from integrating the densities b(L|lambda) of that paper's Table 3. The
fragments placed on a grid (``grid_counts``) are checked against each
class's own b_0 and b_3, against placements derived by hand, and against
the densities integrated over each node's hat function. Sizes
are in um, agglomeration rates in m**3/s and breakage frequencies in 1/s.
The size arrays have different lengths, so a broadcast that drops or swaps
an axis changes the shape.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from PharmaPy.CrystallizationKernels import (BinaryFragmentation, BrownianKernel, ConstantKernel,
                                             ConstantSelection, DifferentialSedimentationKernel,
                                             ErosionFragmentation, ExponentialSelection,
                                             KernelConditions, ParabolicFragmentation,
                                             PowerLawSelection, ProductKernel, ShearKernel,
                                             SumKernel, UniformFragmentation)
from PharmaPy.Errors import PharmaPyValueError

pytestmark = pytest.mark.unit

METRE_PER_MICRON = 1e-6  # [m/um], exact
BOLTZMANN_CONSTANT = 1.380649e-23  # [J/K], exact in the 2019 SI
STANDARD_GRAVITY = 9.80665  # [m/s**2], a defined value
VISCOSITY = 1e-3  # [Pa s], of the order of water's
DENSITY_DIFFERENCE = 300.0  # [kg/m**3], crystal minus liquid
# A sphere of diameter L has volume (pi/6) L**3; unequal to one, so a kernel
# that leaves out the shape factor is caught.
SPHERE_SHAPE_FACTOR = np.pi / 6  # [-]
CONDITIONS = KernelConditions(temp=298.15, supersaturation=0.4, growth_rate=0.1,
                              shape_factor=SPHERE_SHAPE_FACTOR)

SIZES_I = np.array([[5.0], [60.0], [310.0]])  # [um], shape (3, 1)
SIZES_J = np.array([[20.0, 150.0]])  # [um], shape (1, 2)
# Products of the scales the kernels multiply; one double rounding apart.
ALGEBRAIC_RTOL = 1e-12  # [-]


def _volume(size):
    """Crystal volume from its size.

    Parameters
    ----------
    size : numpy.ndarray
        Crystal size [um].

    Returns
    -------
    numpy.ndarray
        Volume [m**3] of a sphere-shaped crystal of that size.
    """
    return SPHERE_SHAPE_FACTOR * (size * METRE_PER_MICRON) ** 3


def test_constant_kernel_is_the_same_for_every_pair():
    rate_constant = 2.5e-14  # [m**3/s]

    rates = ConstantKernel(rate_constant)(SIZES_I, SIZES_J, CONDITIONS)

    np.testing.assert_array_equal(rates, np.full((3, 2), rate_constant))


def test_sum_kernel_scales_with_the_summed_crystal_volume():
    rate_constant = 0.3  # [1/s]

    rates = SumKernel(rate_constant)(SIZES_I, SIZES_J, CONDITIONS)

    expected = rate_constant * (_volume(SIZES_I) + _volume(SIZES_J))
    np.testing.assert_allclose(rates, expected, rtol=ALGEBRAIC_RTOL)


def test_product_kernel_scales_with_the_product_of_crystal_volumes():
    rate_constant = 4e13  # [1/(m**3 s)]

    rates = ProductKernel(rate_constant)(SIZES_I, SIZES_J, CONDITIONS)

    expected = rate_constant * _volume(SIZES_I) * _volume(SIZES_J)
    np.testing.assert_allclose(rates, expected, rtol=ALGEBRAIC_RTOL)


def test_shear_kernel_is_smoluchowskis_collision_rate_times_the_efficiency():
    shear_rate, efficiency = 50.0, 0.3  # [1/s], [-]

    rates = ShearKernel(shear_rate, efficiency)(SIZES_I, SIZES_J, CONDITIONS)

    # (4/3) gamma (a_i + a_j)**3 with the radii a = L/2 in metres.
    radius_i = SIZES_I / 2 * METRE_PER_MICRON  # [m]
    radius_j = SIZES_J / 2 * METRE_PER_MICRON  # [m]
    expected = efficiency * 4 / 3 * shear_rate * (radius_i + radius_j) ** 3
    np.testing.assert_allclose(rates, expected, rtol=ALGEBRAIC_RTOL)


def test_brownian_kernel_is_smoluchowskis_perikinetic_rate():
    rates = BrownianKernel(VISCOSITY)(SIZES_I, SIZES_J, CONDITIONS)

    # 4 pi (D_i + D_j)(a_i + a_j), with the Stokes-Einstein diffusivity
    # D = k_B T / (6 pi mu a) of a sphere of radius a = L/2 in metres.
    radius_i = SIZES_I / 2 * METRE_PER_MICRON  # [m]
    radius_j = SIZES_J / 2 * METRE_PER_MICRON  # [m]
    thermal_energy = BOLTZMANN_CONSTANT * CONDITIONS.temp  # [J]
    diffusivity_i = thermal_energy / (6 * np.pi * VISCOSITY * radius_i)  # [m**2/s]
    diffusivity_j = thermal_energy / (6 * np.pi * VISCOSITY * radius_j)  # [m**2/s]
    expected = 4 * np.pi * (diffusivity_i + diffusivity_j) * (radius_i + radius_j)
    np.testing.assert_allclose(rates, expected, rtol=ALGEBRAIC_RTOL)


def test_brownian_kernel_follows_the_temperature_of_the_conditions():
    warmer = CONDITIONS._replace(temp=2 * CONDITIONS.temp)
    kernel = BrownianKernel(VISCOSITY)

    np.testing.assert_allclose(kernel(SIZES_I, SIZES_J, warmer),
                               2 * kernel(SIZES_I, SIZES_J, CONDITIONS),
                               rtol=ALGEBRAIC_RTOL)


def test_brownian_kernel_refuses_crystals_of_zero_size():
    with pytest.raises(PharmaPyValueError, match="rad > 0"):
        BrownianKernel(VISCOSITY)(np.array([[0.0], [60.0]]), SIZES_J, CONDITIONS)


def test_sedimentation_kernel_is_the_rate_at_which_faster_crystals_overtake():
    efficiency = 0.3  # [-]

    rates = DifferentialSedimentationKernel(DENSITY_DIFFERENCE, VISCOSITY, efficiency)(
        SIZES_I, SIZES_J, CONDITIONS)

    # (pi/4)(d_i + d_j)**2 |u_i - u_j|, with the Stokes settling velocity
    # u = g drho d**2 / (18 mu) of a sphere of diameter d in metres.
    diameter_i = SIZES_I * METRE_PER_MICRON  # [m]
    diameter_j = SIZES_J * METRE_PER_MICRON  # [m]
    velocity_i = STANDARD_GRAVITY * DENSITY_DIFFERENCE * diameter_i ** 2 / (18 * VISCOSITY)  # [m/s]
    velocity_j = STANDARD_GRAVITY * DENSITY_DIFFERENCE * diameter_j ** 2 / (18 * VISCOSITY)  # [m/s]
    expected = (efficiency * np.pi / 4 * (diameter_i + diameter_j) ** 2
                * np.abs(velocity_i - velocity_j))
    np.testing.assert_allclose(rates, expected, rtol=ALGEBRAIC_RTOL)


def test_crystals_of_equal_size_never_overtake_each_other():
    kernel = DifferentialSedimentationKernel(DENSITY_DIFFERENCE, VISCOSITY)

    np.testing.assert_array_equal(kernel(SIZES_I, SIZES_I.T, CONDITIONS).diagonal(),
                                  np.zeros(3))


BUILT_IN_KERNELS = {
    "constant": ConstantKernel(2.5e-14),
    "brownian": BrownianKernel(VISCOSITY),
    "sum": SumKernel(0.3),
    "shear": ShearKernel(50.0, 0.3),
    "sedimentation": DifferentialSedimentationKernel(DENSITY_DIFFERENCE, VISCOSITY, 0.3),
    "product": ProductKernel(4e13),
}


@pytest.mark.parametrize("name", BUILT_IN_KERNELS)
def test_kernels_are_symmetric_in_the_two_crystals(name):
    kernel = BUILT_IN_KERNELS[name]

    np.testing.assert_allclose(kernel(SIZES_I, SIZES_J, CONDITIONS),
                               kernel(SIZES_J.T, SIZES_I.T, CONDITIONS).T,
                               rtol=ALGEBRAIC_RTOL)


# Degree gamma of Marchisio et al. (2003, Table 1): beta(aL, a lambda) =
# a**gamma beta(L, lambda). The product kernel, not in that table, has six,
# three for each volume.
@pytest.mark.parametrize("name, degree", [
    ("constant", 0), ("brownian", 0), ("sum", 3), ("shear", 3), ("sedimentation", 4),
    ("product", 6),
])
def test_kernels_scale_with_their_degree_of_homogeneity(name, degree):
    kernel = BUILT_IN_KERNELS[name]
    stretch = 2.7  # [-], any factor that is not one

    np.testing.assert_allclose(kernel(stretch * SIZES_I, stretch * SIZES_J, CONDITIONS),
                               stretch ** degree * kernel(SIZES_I, SIZES_J, CONDITIONS),
                               rtol=ALGEBRAIC_RTOL)


@pytest.mark.parametrize("make_kernel, message", [
    (lambda: ConstantKernel(-1e-14), "rate_constant must be a finite, non-negative"),
    (lambda: SumKernel(np.nan), "rate_constant must be a finite, non-negative"),
    (lambda: ProductKernel(np.inf), "rate_constant must be a finite, non-negative"),
    (lambda: ShearKernel(-5.0), "shear_rate must be a finite, non-negative"),
    (lambda: ShearKernel(5.0, efficiency=-0.1), "efficiency must be a finite, non-negative"),
    (lambda: ShearKernel(5.0, efficiency=1.5), "cannot exceed 1"),
    (lambda: BrownianKernel(0.0), "viscosity must be a finite, positive"),
    (lambda: BrownianKernel(-1e-3), "viscosity must be a finite, positive"),
    (lambda: DifferentialSedimentationKernel(-300.0, VISCOSITY),
     "density_difference must be a finite, non-negative"),
    (lambda: DifferentialSedimentationKernel(DENSITY_DIFFERENCE, np.inf),
     "viscosity must be a finite, positive"),
    (lambda: DifferentialSedimentationKernel(DENSITY_DIFFERENCE, VISCOSITY, 1.5),
     "cannot exceed 1"),
], ids=["negative_constant", "nan_sum", "infinite_product", "negative_shear_rate",
        "negative_efficiency", "efficiency_above_one", "zero_viscosity",
        "negative_viscosity", "negative_density_difference", "infinite_viscosity",
        "sedimentation_efficiency_above_one"])
def test_invalid_kernel_parameters_are_refused(make_kernel, message):
    with pytest.raises(PharmaPyValueError, match=message):
        make_kernel()


PARENT_SIZES = np.array([5.0, 60.0, 310.0])  # [um]
ORDERS = np.arange(6)  # [-], the moments of a three-node QMOM
# quad's own tolerance, 1e-12, with a margin of 100.
INTEGRATION_RTOL = 1e-10  # [-]


def test_constant_selection_is_the_same_for_every_crystal():
    rate_constant = 2e-3  # [1/s]

    rates = ConstantSelection(rate_constant)(PARENT_SIZES, CONDITIONS)

    np.testing.assert_array_equal(rates, np.full(3, rate_constant))


def test_power_law_selection_is_its_rate_constant_at_the_reference_size():
    selection = PowerLawSelection(2e-3, 1.5, reference_size=60.0)  # [1/s], [-], [um]

    rates = selection(PARENT_SIZES, CONDITIONS)

    assert rates[1] == pytest.approx(2e-3, rel=ALGEBRAIC_RTOL)
    # Each size relative to 60 um, raised to the exponent.
    np.testing.assert_allclose(rates / rates[1], (PARENT_SIZES / 60.0) ** 1.5,
                               rtol=ALGEBRAIC_RTOL)


def test_exponential_selection_grows_log_linearly_with_crystal_volume():
    rate_constant, volume_coefficient = 2e-3, 1e-7  # [1/s], [1/um**3]
    selection = ExponentialSelection(rate_constant, volume_coefficient)

    rates = selection(PARENT_SIZES, CONDITIONS)

    assert selection(np.array([0.0]), CONDITIONS)[0] == rate_constant
    # A relative error of 1e-12 in a rate is an absolute 1e-12 in its log,
    # which a relative check of the 1.25e-5 exponent of the 5 um crystal
    # would not allow.
    np.testing.assert_allclose(np.log(rates / rate_constant),
                               volume_coefficient * PARENT_SIZES ** 3, rtol=0.0,
                               atol=ALGEBRAIC_RTOL)


# Each builds one selection function, with or without a minimum size. Rates are
# in 1/s, the exponent is [-], the reference size in um and the coefficient in 1/um**3.
SELECTION_LAWS = {
    "constant": lambda **kwargs: ConstantSelection(2e-3, **kwargs),
    "power_law": lambda **kwargs: PowerLawSelection(2e-3, 1.5, reference_size=60.0, **kwargs),
    "exponential": lambda **kwargs: ExponentialSelection(2e-3, 1e-7, **kwargs),
}
MINIMUM_SIZE = 60.0  # [um], exactly the middle of PARENT_SIZES


@pytest.mark.parametrize("name", SELECTION_LAWS)
def test_crystals_up_to_the_minimum_size_do_not_break(name):
    make_law = SELECTION_LAWS[name]

    rates = make_law(minimum_size=MINIMUM_SIZE)(PARENT_SIZES, CONDITIONS)

    full_rates = make_law()(PARENT_SIZES, CONDITIONS)
    assert full_rates[2] > 0  # guards against zero matching zero
    np.testing.assert_array_equal(rates, [0.0, 0.0, full_rates[2]])


@pytest.mark.parametrize("name", SELECTION_LAWS)
def test_a_minimum_size_below_every_crystal_changes_no_rate(name):
    make_law = SELECTION_LAWS[name]

    rates = make_law(minimum_size=PARENT_SIZES[0] / 2)(PARENT_SIZES, CONDITIONS)

    np.testing.assert_array_equal(rates, make_law()(PARENT_SIZES, CONDITIONS))


@pytest.mark.parametrize("name", ["constant", "exponential"])
def test_without_a_minimum_size_even_a_crystal_of_zero_size_breaks(name):
    # The default is no minimum, not a minimum of zero, so the laws stay exactly
    # as they were before the parameter existed.
    make_law = SELECTION_LAWS[name]

    assert make_law()(np.array([0.0]), CONDITIONS)[0] == 2e-3
    assert make_law(minimum_size=0.0)(np.array([0.0]), CONDITIONS)[0] == 0.0


@pytest.mark.parametrize("name", SELECTION_LAWS)
def test_a_minimum_size_keeps_the_shape_of_the_sizes(name):
    selection = SELECTION_LAWS[name](minimum_size=MINIMUM_SIZE)

    assert selection(SIZES_I, CONDITIONS).shape == SIZES_I.shape
    assert np.shape(selection(MINIMUM_SIZE, CONDITIONS)) == ()
    assert selection(MINIMUM_SIZE, CONDITIONS) == 0.0


def test_a_minimum_size_does_not_hide_a_size_that_is_not_a_number():
    # Zero here would read as a crystal that does not break; NaN reaches the
    # population balance's own check of the frequencies.
    selection = SELECTION_LAWS["power_law"](minimum_size=MINIMUM_SIZE)

    assert np.isnan(selection(np.array([np.nan]), CONDITIONS)[0])


def _moments_of_fragments(fragment_sizes):
    """Moments of each parent's fragments, from their sizes.

    Parameters
    ----------
    fragment_sizes : list of tuple of float
        For each parent, the sizes of its fragments [um].

    Returns
    -------
    numpy.ndarray
        b_k [um**k], shape ``(parents, len(ORDERS))``.
    """
    return np.array([[sum(size ** k for size in sizes) for k in ORDERS]
                     for sizes in fragment_sizes])


@pytest.mark.parametrize("volume_fraction", [0.5, 0.2], ids=["symmetric", "mass_ratio_1_4"])
def test_binary_fragments_split_the_parent_volume(volume_fraction):
    fragments = BinaryFragmentation(volume_fraction)

    moments = fragments.moments(PARENT_SIZES, ORDERS)

    # Fragments of volume f v and (1 - f) v have sizes f**(1/3) L and so on.
    expected = _moments_of_fragments([
        (np.cbrt(volume_fraction * size ** 3), np.cbrt((1 - volume_fraction) * size ** 3))
        for size in PARENT_SIZES])
    np.testing.assert_allclose(moments, expected, rtol=ALGEBRAIC_RTOL)


def test_erosion_chips_one_fragment_off_parents_larger_than_it():
    fragment_size = 10.0  # [um], larger than the 5 um parent

    moments = ErosionFragmentation(fragment_size).moments(PARENT_SIZES, ORDERS)

    expected = _moments_of_fragments([
        (5.0,),  # no larger than the fragment: left whole
        (fragment_size, np.cbrt(60.0 ** 3 - fragment_size ** 3)),
        (fragment_size, np.cbrt(310.0 ** 3 - fragment_size ** 3)),
    ])
    np.testing.assert_allclose(moments, expected, rtol=ALGEBRAIC_RTOL)


def _parabolic_density(size, parent_size, shape_parameter):
    """Fragment density of the parabolic distribution (Marchisio et al., 2003, Table 3).

    Parameters
    ----------
    size : float
        Fragment size L [um].
    parent_size : float
        Parent size lambda [um].
    shape_parameter : float
        C [-].

    Returns
    -------
    float
        b(L|lambda) [1/um].
    """
    ratio = size / parent_size  # [-]
    return (3 * shape_parameter * ratio ** 2
            + (1 - shape_parameter / 2) * (72 * ratio ** 8 - 72 * ratio ** 5 + 18 * ratio ** 2)
            ) / parent_size


def _uniform_density(size, parent_size):
    """Fragment density of the uniform distribution (Marchisio et al., 2003, Table 3).

    Parameters
    ----------
    size : float
        Fragment size L [um].
    parent_size : float
        Parent size lambda [um].

    Returns
    -------
    float
        b(L|lambda) = 6 L**2 / lambda**3 [1/um].
    """
    return 6 * size ** 2 / parent_size ** 3


@pytest.mark.parametrize("fragments, density", [
    (ParabolicFragmentation(0.5), lambda size, parent: _parabolic_density(size, parent, 0.5)),
    (ParabolicFragmentation(2.8), lambda size, parent: _parabolic_density(size, parent, 2.8)),
    (UniformFragmentation(), _uniform_density),
], ids=["parabolic_low_c", "parabolic_high_c", "uniform"])
def test_spread_fragments_match_the_moments_of_their_density(fragments, density):
    moments = fragments.moments(PARENT_SIZES, ORDERS)

    expected = np.array([[quad(lambda size: size ** k * density(size, parent), 0.0, parent,
                               epsabs=0.0, epsrel=1e-12)[0] for k in ORDERS]
                         for parent in PARENT_SIZES])
    np.testing.assert_allclose(moments, expected, rtol=INTEGRATION_RTOL)


def test_parabolic_fragments_with_c_of_two_are_uniform():
    np.testing.assert_allclose(ParabolicFragmentation(2.0).moments(PARENT_SIZES, ORDERS),
                               UniformFragmentation().moments(PARENT_SIZES, ORDERS),
                               rtol=ALGEBRAIC_RTOL)


@pytest.mark.parametrize("fragments, fragment_counts", [
    (BinaryFragmentation(0.5), [2, 2, 2]),
    (BinaryFragmentation(0.2), [2, 2, 2]),
    (ErosionFragmentation(10.0), [1, 2, 2]),  # the 5 um parent is left whole
    (ParabolicFragmentation(1.0), [2, 2, 2]),
    (UniformFragmentation(), [2, 2, 2]),
], ids=["symmetric", "mass_ratio_1_4", "erosion", "parabolic", "uniform"])
def test_fragments_are_counted_and_hold_their_parents_volume(fragments, fragment_counts):
    # The paper's two constraints: b_0 counts the fragments, b_3 is the
    # parent's volume.
    moments = fragments.moments(PARENT_SIZES, ORDERS)

    np.testing.assert_allclose(moments[:, 0], fragment_counts, rtol=ALGEBRAIC_RTOL)
    np.testing.assert_allclose(moments[:, 3], PARENT_SIZES ** 3, rtol=ALGEBRAIC_RTOL)


# Unequal spacing with a first node above zero size, and a grid from zero
# size, below whose first node no fragment can fall.
GEOMETRIC_GRID = np.geomspace(2.0, 400.0, 23)  # [um]
GRID_FROM_ZERO = np.linspace(0.0, 300.0, 16)  # [um]
GRID_FRAGMENTS = {
    "symmetric": BinaryFragmentation(0.5),
    "mass_ratio_1_4": BinaryFragmentation(0.2),
    "erosion": ErosionFragmentation(7.3),  # [um], between nodes of both grids
    # A node of GRID_FROM_ZERO: that parent stays whole, with no empty
    # remainder left at the zero-size node.
    "erosion_at_a_node": ErosionFragmentation(20.0),  # [um]
    "parabolic_low_c": ParabolicFragmentation(0.5),
    "parabolic_high_c": ParabolicFragmentation(2.8),
    "uniform": UniformFragmentation(),
}
SPREAD_DENSITIES = {
    "parabolic_low_c": lambda size, parent: _parabolic_density(size, parent, 0.5),
    "parabolic_high_c": lambda size, parent: _parabolic_density(size, parent, 2.8),
    "uniform": _uniform_density,
}


@pytest.mark.parametrize("grid", [GEOMETRIC_GRID, GRID_FROM_ZERO], ids=["geometric", "from_zero"])
@pytest.mark.parametrize("name", GRID_FRAGMENTS)
def test_fragments_placed_on_a_grid_hold_their_parents_volume(name, grid):
    fragments = GRID_FRAGMENTS[name]

    counts = fragments.grid_counts(grid)

    placed_volume = grid ** 3 @ counts  # [um**3], per parent
    np.testing.assert_allclose(placed_volume, fragments.moments(grid, [3])[:, 0],
                               rtol=ALGEBRAIC_RTOL)
    assert np.all(counts >= 0)


@pytest.mark.parametrize("name", GRID_FRAGMENTS)
def test_every_fragment_is_counted_when_none_falls_below_the_first_node(name):
    fragments = GRID_FRAGMENTS[name]

    counts = fragments.grid_counts(GRID_FROM_ZERO)

    np.testing.assert_allclose(counts.sum(axis=0), fragments.moments(GRID_FROM_ZERO, [0])[:, 0],
                               rtol=ALGEBRAIC_RTOL)


@pytest.mark.parametrize("name", SPREAD_DENSITIES)
def test_fragments_below_the_first_node_keep_their_volume_there(name):
    # A fragment of volume w below the first node's v_0 counts as w / v_0
    # crystals there, so each parent's fragments fall short of b_0 by the
    # integral of b(L|lambda) (1 - L**3 / L_0**3) up to the first node L_0.
    density = SPREAD_DENSITIES[name]
    first_node = GEOMETRIC_GRID[0]  # [um]

    counts = GRID_FRAGMENTS[name].grid_counts(GEOMETRIC_GRID)

    shortfall = np.array([quad(lambda size: density(size, parent) * (1 - (size / first_node) ** 3),
                               0.0, first_node, epsabs=0.0, epsrel=1e-12)[0]
                          for parent in GEOMETRIC_GRID])  # [-]
    np.testing.assert_allclose(counts.sum(axis=0), 2 - shortfall, rtol=ALGEBRAIC_RTOL)


@pytest.mark.parametrize("name", GRID_FRAGMENTS)
def test_breaking_the_first_nodes_crystals_leaves_them_as_they_are(name):
    # Every fragment of a parent at the first node falls below it, and keeps
    # its volume there: one crystal of the parent's volume.
    counts = GRID_FRAGMENTS[name].grid_counts(GEOMETRIC_GRID)

    expected = np.zeros(len(GEOMETRIC_GRID))
    expected[0] = 1.0
    np.testing.assert_allclose(counts[:, 0], expected, rtol=0, atol=ALGEBRAIC_RTOL)


# Each node holds twice the volume of the one below, so fragments of half a
# parent's volume land on a node.
DOUBLING_GRID = 10.0 * 2.0 ** (np.arange(6) / 3)  # [um]


def test_halves_of_a_parent_land_on_the_node_below_it():
    counts = BinaryFragmentation(0.5).grid_counts(DOUBLING_GRID)

    expected = 2 * np.eye(len(DOUBLING_GRID), k=1)
    expected[0, 0] = 1.0  # both halves below the first node keep its volume
    np.testing.assert_allclose(counts, expected, rtol=0, atol=ALGEBRAIC_RTOL)


def test_eroded_parents_are_split_between_the_nodes_around_them():
    # The fragment has node 2's volume v_2. Parents up to node 2 stay whole;
    # node 3's (2 v_2) leaves v_2, on node 2; node 4's (4 v_2) leaves 3 v_2,
    # midway between nodes 3 and 4 in volume; node 5's (8 v_2) leaves 7 v_2,
    # three quarters of the way from node 4 to node 5.
    counts = ErosionFragmentation(DOUBLING_GRID[2]).grid_counts(DOUBLING_GRID)

    expected = np.zeros((6, 6))
    expected[[0, 1, 2], [0, 1, 2]] = 1.0
    expected[2, 3] = 2.0
    expected[[2, 3, 4], [4, 4, 4]] = 1.0, 0.5, 0.5
    expected[[2, 4, 5], [5, 5, 5]] = 1.0, 0.25, 0.75
    np.testing.assert_allclose(counts, expected, rtol=0, atol=ALGEBRAIC_RTOL)


def _hat(node, volume, node_volumes):
    """Share of node ``node`` of a fragment of the given volume, by fixed pivot.

    Parameters
    ----------
    node : int
        Node index.
    volume : float
        Fragment volume L**3 [um**3].
    node_volumes : numpy.ndarray
        Node volumes [um**3], shape ``(n,)``.

    Returns
    -------
    float
        The hat function, linear in volume between neighbouring nodes, one at
        the node and zero at its neighbours; below the first node it falls
        to zero at zero volume.
    """
    unit = np.zeros(len(node_volumes) + 1)
    unit[node + 1] = 1.0
    return np.interp(volume, np.concatenate(([0.0], node_volumes)), unit)


@pytest.mark.parametrize("name", SPREAD_DENSITIES)
def test_spread_fragments_are_their_density_integrated_over_each_nodes_hat(name):
    density = SPREAD_DENSITIES[name]
    grid = np.array([3.0, 7.0, 12.0, 20.0, 33.0])  # [um], neither uniform nor geometric

    counts = GRID_FRAGMENTS[name].grid_counts(grid)

    expected = np.zeros((len(grid), len(grid)))
    for parent, parent_size in enumerate(grid):
        for node in range(parent + 1):
            expected[node, parent] = quad(
                lambda size: density(size, parent_size) * _hat(node, size ** 3, grid ** 3),
                0.0, parent_size, points=grid[:parent] if parent else None,
                epsabs=0.0, epsrel=1e-12)[0]
    np.testing.assert_allclose(counts, expected, rtol=INTEGRATION_RTOL)


@pytest.mark.parametrize("grid", [
    [5.0], [5.0, 3.0, 8.0], [-1.0, 4.0], [1.0, np.nan], [[1.0, 2.0]],
], ids=["one_node", "not_increasing", "negative", "nan", "two_dimensional"])
@pytest.mark.parametrize("fragments", [BinaryFragmentation(), UniformFragmentation()],
                         ids=["placed", "spread"])
def test_grids_that_are_not_size_grids_are_refused(fragments, grid):
    with pytest.raises(PharmaPyValueError, match="grid_sizes must be"):
        fragments.grid_counts(grid)


@pytest.mark.parametrize("make_law, message", [
    (lambda: ConstantSelection(-1e-3), "rate_constant must be a finite, non-negative"),
    (lambda: PowerLawSelection(1e-3, -0.5), "exponent must be a finite, non-negative"),
    (lambda: PowerLawSelection(1e-3, 1.0, reference_size=0.0),
     "reference_size must be a finite, positive"),
    (lambda: ExponentialSelection(1e-3, -1e-7), "volume_coefficient must be a finite, non-negative"),
    (lambda: ConstantSelection(1e-3, minimum_size=-1.0), "minimum_size must be a finite, non-negative"),
    (lambda: PowerLawSelection(1e-3, 1.0, minimum_size=np.nan),
     "minimum_size must be a finite, non-negative"),
    (lambda: ExponentialSelection(1e-3, 1e-7, minimum_size=np.inf),
     "minimum_size must be a finite, non-negative"),
    (lambda: BinaryFragmentation(0.0), "strictly between 0 and 1"),
    (lambda: BinaryFragmentation(1.0), "strictly between 0 and 1"),
    (lambda: ErosionFragmentation(0.0), "fragment_size must be a finite, positive"),
    (lambda: ParabolicFragmentation(-0.1), r"must lie in \[0, 3\]"),
    (lambda: ParabolicFragmentation(3.5), r"must lie in \[0, 3\]"),
], ids=["negative_rate", "shattering_exponent", "zero_reference_size",
        "negative_volume_coefficient", "negative_minimum_size", "nan_minimum_size",
        "infinite_minimum_size", "no_volume", "all_volume", "zero_fragment",
        "parabolic_below", "parabolic_above"])
def test_invalid_breakage_parameters_are_refused(make_law, message):
    with pytest.raises(PharmaPyValueError, match=message):
        make_law()
