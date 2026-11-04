"""Agglomeration kernels for the QMOM population balance.

A kernel is a callable ``kernel(size_i, size_j, conditions)`` returning the
agglomeration rate coefficient beta [m**3/s] of a crystal of size L_i with
one of size L_j [um], per crystal pair per m**3 of slurry, broadcast over
the two size arrays. ``conditions`` is a ``KernelConditions``. A user kernel
is any callable with this signature, returning finite, non-negative rates;
one that is infinite at zero size sets ``needs_finite_nuclei = True``, and
the population balance then asks for nuclei born at a finite size.

The built-in kernels follow Table 1 of Marchisio, Vigil and Fox (2003, J.
Colloid Interface Sci. 258:322-334), with physical prefactors: constant,
Brownian, sum, hydrodynamic (``ShearKernel``) and differential force
(``DifferentialSedimentationKernel``), plus ``ProductKernel``.
"""

from typing import NamedTuple

import numpy as np

from PharmaPy.Errors import PharmaPyValueError

# Exact unit conversions of a crystal size and volume.
METRE_PER_MICRON = 1e-6  # [m/um]
CUBIC_METRE_PER_CUBIC_MICRON = 1e-18  # [m**3/um**3]
# Exact in the 2019 SI.
BOLTZMANN_CONSTANT = 1.380649e-23  # [J/K]
# Standard acceleration of gravity, a defined value (3rd CGPM, 1901).
STANDARD_GRAVITY = 9.80665  # [m/s**2]


class KernelConditions(NamedTuple):
    """State a kernel may depend on, evaluated at each right-hand side.

    Attributes
    ----------
    temp : float
        Liquid temperature [K].
    supersaturation : float
        Driving force of the vessel's ``CrystKinetics``, in its
        ``sup_sat_type`` [-] or [kg/m**3].
    growth_rate : float
        Size-independent growth rate G0 [um/s], zero while dissolving.
    shape_factor : float
        Volume shape factor kv [-]; a crystal of size L has volume kv L**3.
    """

    temp: float
    supersaturation: float
    growth_rate: float
    shape_factor: float


def _validated_rate_constant(value, name):
    """Check a kernel parameter is a finite, non-negative number.

    Parameters
    ----------
    value : float
        Parameter value, in the kernel's units.
    name : str
        Parameter name, for the error message.

    Returns
    -------
    float
        ``value`` as a float.

    Raises
    ------
    PharmaPyValueError
        If ``value`` is negative, NaN or infinite.
    """
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise PharmaPyValueError(
            '%s must be a finite, non-negative number, got %r.' % (name, value))
    return value


def _validated_positive(value, name):
    """Check a kernel parameter is a finite, positive number.

    Parameters
    ----------
    value : float
        Parameter value, in the kernel's units.
    name : str
        Parameter name, for the error message.

    Returns
    -------
    float
        ``value`` as a float.

    Raises
    ------
    PharmaPyValueError
        If ``value`` is zero, negative, NaN or infinite.
    """
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise PharmaPyValueError(
            '%s must be a finite, positive number, got %r.' % (name, value))
    return value


def _validated_efficiency(efficiency):
    """Check a collision efficiency lies in [0, 1].

    Parameters
    ----------
    efficiency : float
        Fraction of collisions that agglomerate [-].

    Returns
    -------
    float
        ``efficiency`` as a float.

    Raises
    ------
    PharmaPyValueError
        If ``efficiency`` is negative, above one or not finite.
    """
    efficiency = _validated_rate_constant(efficiency, 'efficiency')  # [-]
    if efficiency > 1:
        raise PharmaPyValueError(
            'efficiency is a fraction of collisions and cannot exceed 1, '
            'got %r.' % efficiency)
    return efficiency


def _crystal_volume(size, shape_factor):
    """Volume of a crystal from its size.

    Parameters
    ----------
    size : numpy.ndarray
        Crystal size L [um].
    shape_factor : float
        Volume shape factor kv [-].

    Returns
    -------
    numpy.ndarray
        kv L**3 [m**3], shaped like ``size``.
    """
    return shape_factor * np.asarray(size, dtype=float) ** 3 * CUBIC_METRE_PER_CUBIC_MICRON


class ConstantKernel:
    """Size-independent agglomeration, beta = beta_0.

    Smoluchowski's constant kernel, with a closed-form solution for any
    initial distribution (Scott, 1968, J. Atmos. Sci. 25:54-65).

    Parameters
    ----------
    rate_constant : float
        beta_0 [m**3/s], at least zero.

    Raises
    ------
    PharmaPyValueError
        If ``rate_constant`` is negative or not finite.
    """

    def __init__(self, rate_constant):
        self.rate_constant = _validated_rate_constant(rate_constant, 'rate_constant')  # [m**3/s]

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um], broadcast against each other.
        conditions : KernelConditions
            Unused.

        Returns
        -------
        numpy.ndarray
            beta_0 [m**3/s], with the broadcast shape of the sizes.
        """
        return np.full(np.broadcast(size_i, size_j).shape, self.rate_constant)


class BrownianKernel:
    """Perikinetic agglomeration, beta = (2 k_B T / 3 mu) (L_i + L_j)**2 / (L_i L_j).

    Smoluchowski's (1917, Z. Phys. Chem. 92:129-168) rate of Brownian
    collisions, 4 pi (D_i + D_j)(a_i + a_j), for spheres of radii a = L/2
    with Stokes-Einstein diffusivities D = k_B T / (6 pi mu a), the
    temperature T taken from the kernel conditions. For equal sizes it is
    8 k_B T / (3 mu) against ``ShearKernel``'s (4/3) gamma L**3, which takes
    over above L**3 = 2 k_B T / (mu gamma), about 0.4 um in water at
    gamma = 100 1/s. It grows without bound as either size goes to zero, so
    it needs nuclei born at a finite size.

    Parameters
    ----------
    viscosity : float
        Dynamic viscosity of the liquid mu [Pa s], above zero.

    Raises
    ------
    PharmaPyValueError
        If ``viscosity`` is not finite and positive.
    """

    # Infinite at zero size: QMOM asks for a nucleus size rad > 0.
    needs_finite_nuclei = True

    def __init__(self, viscosity):
        self.viscosity = _validated_positive(viscosity, 'viscosity')  # [Pa s]

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um], broadcast against each other, all positive.
        conditions : KernelConditions
            Supplies the temperature.

        Returns
        -------
        numpy.ndarray
            (2 k_B T / 3 mu) (L_i + L_j)**2 / (L_i L_j) [m**3/s].

        Raises
        ------
        PharmaPyValueError
            If any size is zero or negative, as for nuclei born at
            rad = 0, where the kernel has no finite value.
        """
        size_i = np.asarray(size_i, dtype=float)
        size_j = np.asarray(size_j, dtype=float)

        if np.any(size_i <= 0) or np.any(size_j <= 0):
            raise PharmaPyValueError(
                'BrownianKernel is infinite at zero size, where nuclei born '
                'at rad = 0 sit. Give the population balance a nucleus size '
                'rad > 0.')

        thermal_rate = 2 * BOLTZMANN_CONSTANT * conditions.temp / (3 * self.viscosity)  # [m**3/s]
        return thermal_rate * (size_i + size_j) ** 2 / (size_i * size_j)


class SumKernel:
    """Agglomeration proportional to the summed volume, beta = beta_0 (v_i + v_j).

    Golovin's kernel, with v = kv L**3 the crystal volume; closed-form for
    any initial distribution (Scott, 1968, J. Atmos. Sci. 25:54-65). It is
    the "sum" kernel of Marchisio et al. (2003), often used to approximate
    the hydrodynamic one.

    Parameters
    ----------
    rate_constant : float
        beta_0 [1/s], at least zero.

    Raises
    ------
    PharmaPyValueError
        If ``rate_constant`` is negative or not finite.
    """

    def __init__(self, rate_constant):
        self.rate_constant = _validated_rate_constant(rate_constant, 'rate_constant')  # [1/s]

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um], broadcast against each other.
        conditions : KernelConditions
            Supplies the shape factor.

        Returns
        -------
        numpy.ndarray
            beta_0 (v_i + v_j) [m**3/s].
        """
        kv = conditions.shape_factor
        return self.rate_constant * (_crystal_volume(size_i, kv)
                                     + _crystal_volume(size_j, kv))


class ProductKernel:
    """Agglomeration proportional to the volume product, beta = beta_0 v_i v_j.

    With v = kv L**3 the crystal volume. The population gels in finite time,
    after which the moments no longer describe it (Scott, 1968, J. Atmos.
    Sci. 25:54-65).

    Parameters
    ----------
    rate_constant : float
        beta_0 [1/(m**3 s)], at least zero.

    Raises
    ------
    PharmaPyValueError
        If ``rate_constant`` is negative or not finite.
    """

    def __init__(self, rate_constant):
        self.rate_constant = _validated_rate_constant(rate_constant, 'rate_constant')  # [1/(m**3 s)]

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um], broadcast against each other.
        conditions : KernelConditions
            Supplies the shape factor.

        Returns
        -------
        numpy.ndarray
            beta_0 v_i v_j [m**3/s].
        """
        kv = conditions.shape_factor
        return self.rate_constant * _crystal_volume(size_i, kv) * _crystal_volume(size_j, kv)


class ShearKernel:
    """Orthokinetic agglomeration in laminar shear, beta = (e gamma / 6) (L_i + L_j)**3.

    Smoluchowski's (1917, Z. Phys. Chem. 92:129-168) collision rate
    (4/3) gamma (a_i + a_j)**3 for spheres of radii a = L/2, times a collision
    efficiency e: the "hydrodynamic" kernel (L_i + L_j)**3 of Marchisio et al.
    (2003). The crystal size is taken as the collision diameter.

    Parameters
    ----------
    shear_rate : float
        Shear rate gamma [1/s], at least zero.
    efficiency : float, optional
        Fraction of collisions that agglomerate, e [-], between 0 and 1.
        Defaults to 1, Smoluchowski's assumption that every collision sticks.

    Raises
    ------
    PharmaPyValueError
        If ``shear_rate`` is negative or not finite, or ``efficiency`` lies
        outside [0, 1].
    """

    def __init__(self, shear_rate, efficiency=1.0):
        self.shear_rate = _validated_rate_constant(shear_rate, 'shear_rate')  # [1/s]
        self.efficiency = _validated_efficiency(efficiency)  # [-]

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um], broadcast against each other.
        conditions : KernelConditions
            Unused.

        Returns
        -------
        numpy.ndarray
            (e gamma / 6) (L_i + L_j)**3 [m**3/s].
        """
        collision_diameter = np.asarray(size_i, dtype=float) + np.asarray(size_j, dtype=float)  # [um]
        return (self.efficiency * self.shear_rate / 6 * collision_diameter ** 3
                * CUBIC_METRE_PER_CUBIC_MICRON)


class DifferentialSedimentationKernel:
    """Agglomeration by settling, beta = e (pi g drho / 72 mu) (L_i + L_j)**2 |L_i**2 - L_j**2|.

    A crystal settling faster overtakes a slower one: the rectilinear
    collision rate (pi/4)(L_i + L_j)**2 |u_i - u_j|, with the Stokes settling
    velocities u = g drho L**2 / (18 mu), times a collision efficiency e.
    This is the "differential force" kernel of Marchisio et al. (2003).
    Valid while Stokes drag holds, at particle Reynolds numbers well below
    one. Crystals of equal size never meet.

    Parameters
    ----------
    density_difference : float
        |rho_crystal - rho_liquid| drho [kg/m**3], at least zero.
    viscosity : float
        Dynamic viscosity of the liquid mu [Pa s], above zero.
    efficiency : float, optional
        Fraction of collisions that agglomerate, e [-], between 0 and 1.
        Defaults to 1.

    Raises
    ------
    PharmaPyValueError
        If ``density_difference`` is negative or not finite, ``viscosity``
        is not finite and positive, or ``efficiency`` lies outside [0, 1].
    """

    def __init__(self, density_difference, viscosity, efficiency=1.0):
        self.density_difference = _validated_rate_constant(
            density_difference, 'density_difference')  # [kg/m**3]
        self.viscosity = _validated_positive(viscosity, 'viscosity')  # [Pa s]
        self.efficiency = _validated_efficiency(efficiency)  # [-]

    def __call__(self, size_i, size_j, conditions):
        """Agglomeration rate coefficient.

        Parameters
        ----------
        size_i, size_j : numpy.ndarray
            Crystal sizes [um], broadcast against each other.
        conditions : KernelConditions
            Unused.

        Returns
        -------
        numpy.ndarray
            e (pi g drho / 72 mu) (L_i + L_j)**2 |L_i**2 - L_j**2| [m**3/s].
        """
        size_i = np.asarray(size_i, dtype=float) * METRE_PER_MICRON  # [m]
        size_j = np.asarray(size_j, dtype=float) * METRE_PER_MICRON  # [m]
        settling_rate = (np.pi * STANDARD_GRAVITY * self.density_difference
                         / (72 * self.viscosity))  # [1/(m s)]
        return (self.efficiency * settling_rate * (size_i + size_j) ** 2
                * np.abs(size_i ** 2 - size_j ** 2))
