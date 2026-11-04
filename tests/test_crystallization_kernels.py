"""Built-in agglomeration kernels of ``PharmaPy.CrystallizationKernels``.

Expected rates are built from the crystal sizes converted to metres first,
the shear and Brownian kernels from Smoluchowski's radius forms and the
settling kernel from Stokes velocities, so none repeats the kernels' own
expressions; the degrees of homogeneity are those of Table 1 of Marchisio,
Vigil and Fox (2003). Sizes are in um and rates in m**3/s. The size arrays
have different lengths, so a broadcast that drops or swaps an axis changes
the shape.
"""

import numpy as np
import pytest

from PharmaPy.CrystallizationKernels import (BrownianKernel, ConstantKernel,
                                             DifferentialSedimentationKernel, KernelConditions,
                                             ProductKernel, ShearKernel, SumKernel)
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
