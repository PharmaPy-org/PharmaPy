"""Agglomeration and breakage kernels for the QMOM population balance.

An agglomeration kernel is a callable ``kernel(size_i, size_j, conditions)``
returning the rate coefficient beta [m**3/s] of a crystal of size L_i with
one of size L_j [um], per crystal pair per m**3 of slurry, broadcast over
the two size arrays. A breakage selection function is a callable
``selection(size, conditions)`` returning the frequency a(L) [1/s] at which
a crystal of size L breaks. ``conditions`` is a ``KernelConditions``. A
fragment distribution has ``moments(parent_sizes, orders)``, the moments
b_k(L) [um**k] of the fragments of one parent of each size. User versions
are any objects of these shapes; a kernel or selection function that is
infinite at zero size sets ``needs_finite_nuclei = True``, and the
population balance then asks for nuclei born at a finite size.

The built-in kernels follow Tables 1 to 3 of Marchisio, Vigil and Fox
(2003, J. Colloid Interface Sci. 258:322-334), with physical prefactors.
Agglomeration: constant, Brownian, sum, hydrodynamic (``ShearKernel``) and
differential force (``DifferentialSedimentationKernel``), plus
``ProductKernel``. Selection: constant, power law and exponential, each with
an optional size up to which crystals do not break, as in that paper's Table 4.
Fragments: symmetric and 1:4 (``BinaryFragmentation``), erosion, parabolic
and uniform.
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


# ---------------------------------------------------------------------------
# Breakage: selection functions a(L) [1/s] (Marchisio et al., 2003, Table 2)
# ---------------------------------------------------------------------------

def _validated_minimum_size(minimum_size):
    """Check the size up to which crystals do not break, if one is given.

    Parameters
    ----------
    minimum_size : float or None
        Size [um], or None for no such size.

    Returns
    -------
    float or None
        ``minimum_size`` as a float, or None.

    Raises
    ------
    PharmaPyValueError
        If ``minimum_size`` is negative, NaN or infinite.
    """
    if minimum_size is None:
        return None
    return _validated_rate_constant(minimum_size, 'minimum_size')


def _unbroken_up_to(rates, size, minimum_size):
    """Set the breakage frequency to zero up to a minimum size.

    Parameters
    ----------
    rates : numpy.ndarray
        Breakage frequencies [1/s], shaped like ``size``.
    size : numpy.ndarray
        Crystal sizes [um].
    minimum_size : float or None
        Size [um] at and below which crystals do not break, or None.

    Returns
    -------
    numpy.ndarray
        The frequencies [1/s], zero at and below ``minimum_size``.
    """
    if minimum_size is None:
        return rates
    return np.where(np.asarray(size, dtype=float) <= minimum_size, 0.0, rates)


class ConstantSelection:
    """Every crystal breaks at the same frequency, a = S_0.

    Parameters
    ----------
    rate_constant : float
        S_0 [1/s], at least zero.
    minimum_size : float, optional
        Crystals of this size [um] or smaller do not break, like the primary
        particle in Table 4 of Marchisio et al. (2003). At least zero. Defaults
        to None, so every crystal breaks.

    Raises
    ------
    PharmaPyValueError
        If ``rate_constant`` or ``minimum_size`` is negative or not finite.
    """

    def __init__(self, rate_constant, minimum_size=None):
        self.rate_constant = _validated_rate_constant(rate_constant, 'rate_constant')  # [1/s]
        self.minimum_size = _validated_minimum_size(minimum_size)  # [um]

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
            S_0 [1/s], shaped like ``size``, and zero up to ``minimum_size``.
        """
        return _unbroken_up_to(np.full(np.shape(size), self.rate_constant), size,
                               self.minimum_size)


class PowerLawSelection:
    """Larger crystals break more often, a = S_0 (L / L_ref)**alpha.

    Parameters
    ----------
    rate_constant : float
        S_0 [1/s], the frequency at the reference size, at least zero.
    exponent : float
        alpha [-], at least zero. A negative exponent makes small crystals
        break faster, the regime in which breakage can shatter mass into
        crystals of vanishing size (Marchisio et al., 2003), which the
        moments cannot carry; the frequency would also be infinite at zero
        size.
    reference_size : float, optional
        L_ref [um], above zero. Defaults to 1 um, so S_0 is the frequency of
        a one-micron crystal.
    minimum_size : float, optional
        Crystals of this size [um] or smaller do not break, like the primary
        particle in Table 4 of Marchisio et al. (2003). At least zero. Defaults
        to None, so every crystal breaks.

    Raises
    ------
    PharmaPyValueError
        If ``rate_constant``, ``exponent`` or ``minimum_size`` is negative or
        not finite, or ``reference_size`` is not finite and positive.
    """

    def __init__(self, rate_constant, exponent, reference_size=1.0, minimum_size=None):
        self.rate_constant = _validated_rate_constant(rate_constant, 'rate_constant')  # [1/s]
        self.exponent = _validated_rate_constant(exponent, 'exponent')  # [-]
        self.reference_size = _validated_positive(reference_size, 'reference_size')  # [um]
        self.minimum_size = _validated_minimum_size(minimum_size)  # [um]

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
            S_0 (L / L_ref)**alpha [1/s], shaped like ``size``, and zero up to
            ``minimum_size``.
        """
        rates = self.rate_constant * (np.asarray(size, dtype=float)
                                      / self.reference_size) ** self.exponent  # [1/s]
        return _unbroken_up_to(rates, size, self.minimum_size)


class ExponentialSelection:
    """Breakage frequency growing with crystal volume, a = S_0 exp(delta L**3).

    Parameters
    ----------
    rate_constant : float
        S_0 [1/s], the frequency as L goes to zero, at least zero.
    volume_coefficient : float
        delta [1/um**3], at least zero, so that, as for the power law, small
        crystals do not break faster than large ones.
    minimum_size : float, optional
        Crystals of this size [um] or smaller do not break, like the primary
        particle in Table 4 of Marchisio et al. (2003). At least zero. Defaults
        to None, so every crystal breaks.

    Raises
    ------
    PharmaPyValueError
        If any parameter is negative or not finite.
    """

    def __init__(self, rate_constant, volume_coefficient, minimum_size=None):
        self.rate_constant = _validated_rate_constant(rate_constant, 'rate_constant')  # [1/s]
        self.volume_coefficient = _validated_rate_constant(
            volume_coefficient, 'volume_coefficient')  # [1/um**3]
        self.minimum_size = _validated_minimum_size(minimum_size)  # [um]

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
            S_0 exp(delta L**3) [1/s], shaped like ``size``, and zero up to
            ``minimum_size``.
        """
        rates = self.rate_constant * np.exp(
            self.volume_coefficient * np.asarray(size, dtype=float) ** 3)  # [1/s]
        return _unbroken_up_to(rates, size, self.minimum_size)


# ---------------------------------------------------------------------------
# Breakage: fragment distributions (Marchisio et al., 2003, Table 3)
# ---------------------------------------------------------------------------

def _size_powers(parent_sizes, orders):
    """Parent sizes raised to each order.

    Parameters
    ----------
    parent_sizes : numpy.ndarray
        Parent sizes L [um], shape ``(n,)``.
    orders : numpy.ndarray
        Moment orders k, shape ``(m,)``.

    Returns
    -------
    numpy.ndarray
        L**k [um**k], shape ``(n, m)``.
    """
    return np.asarray(parent_sizes, dtype=float)[:, None] ** np.asarray(orders)[None, :]


class BinaryFragmentation:
    """Two fragments holding the fractions f and 1 - f of the parent's volume.

    b_k(L) = (f**(k/3) + (1 - f)**(k/3)) L**k. The paper's symmetric
    fragmentation is f = 1/2, the default, and its mass ratio 1:4 is f = 1/5.

    Parameters
    ----------
    volume_fraction : float, optional
        f [-], strictly between 0 and 1. Defaults to 0.5.

    Raises
    ------
    PharmaPyValueError
        If ``volume_fraction`` is not strictly between 0 and 1.
    """

    def __init__(self, volume_fraction=0.5):
        volume_fraction = float(volume_fraction)
        if not 0 < volume_fraction < 1:
            raise PharmaPyValueError(
                'volume_fraction must lie strictly between 0 and 1, so that both '
                'fragments exist, got %r.' % volume_fraction)
        self.volume_fraction = volume_fraction  # [-]

    def moments(self, parent_sizes, orders):
        """Moments of the two fragments of each parent.

        Parameters
        ----------
        parent_sizes : numpy.ndarray
            Parent sizes L [um], shape ``(n,)``.
        orders : numpy.ndarray
            Moment orders k, shape ``(m,)``.

        Returns
        -------
        numpy.ndarray
            b_k(L) [um**k], shape ``(n, m)``.
        """
        thirds = np.asarray(orders) / 3
        share = self.volume_fraction ** thirds + (1 - self.volume_fraction) ** thirds  # [-]
        return share[None, :] * _size_powers(parent_sizes, orders)


class ErosionFragmentation:
    """A small fragment of fixed size chipped off the parent.

    b_k(L) = L_0**k + (L**3 - L_0**3)**(k/3): one fragment of size L_0 and
    the eroded parent, which keeps the rest of the volume. A parent no
    larger than L_0 has nothing to lose and does not erode: b_k(L) = L**k.

    Parameters
    ----------
    fragment_size : float
        L_0 [um], above zero; the paper's primary particle.

    Raises
    ------
    PharmaPyValueError
        If ``fragment_size`` is not finite and positive.
    """

    def __init__(self, fragment_size):
        self.fragment_size = _validated_positive(fragment_size, 'fragment_size')  # [um]

    def moments(self, parent_sizes, orders):
        """Moments of the fragment and the eroded parent.

        Parameters
        ----------
        parent_sizes : numpy.ndarray
            Parent sizes L [um], shape ``(n,)``.
        orders : numpy.ndarray
            Moment orders k, shape ``(m,)``.

        Returns
        -------
        numpy.ndarray
            b_k(L) [um**k], shape ``(n, m)``.
        """
        parent_sizes = np.asarray(parent_sizes, dtype=float)
        thirds = np.asarray(orders) / 3
        erodes = parent_sizes > self.fragment_size
        remaining_cube = np.where(erodes, parent_sizes ** 3 - self.fragment_size ** 3, 0.0)  # [um**3]
        eroded = (self.fragment_size ** np.asarray(orders))[None, :] + remaining_cube[:, None] ** thirds
        return np.where(erodes[:, None], eroded, _size_powers(parent_sizes, orders))


class ParabolicFragmentation:
    """Two fragments on average, spread by a parabola-like law in size.

    b(L|lambda) = 3C L**2/lambda**3 + (1 - C/2)(72 L**8/lambda**9
    - 72 L**5/lambda**6 + 18 L**2/lambda**3) on 0 < L < lambda, so
    b_k = (3C/(k+3) + (1 - C/2)(72/(k+9) - 72/(k+6) + 18/(k+3))) lambda**k.
    C = 2 is the uniform distribution.

    Parameters
    ----------
    shape_parameter : float
        C [-], from 0 to 3. Written as b = (x**2/lambda)(3C + 18 (1 - C/2)
        (2x**3 - 1)**2) with x = L/lambda, the bracket is linear in
        (2x**3 - 1)**2, which runs from 0 to 1; it stays non-negative at both
        ends, 3C and 18 - 6C, only for 0 <= C <= 3.

    Raises
    ------
    PharmaPyValueError
        If ``shape_parameter`` lies outside [0, 3].
    """

    def __init__(self, shape_parameter):
        shape_parameter = float(shape_parameter)
        if not 0 <= shape_parameter <= 3:
            raise PharmaPyValueError(
                'shape_parameter must lie in [0, 3], where the fragment '
                'distribution is non-negative, got %r.' % shape_parameter)
        self.shape_parameter = shape_parameter  # [-]

    def moments(self, parent_sizes, orders):
        """Moments of the fragments of each parent.

        Parameters
        ----------
        parent_sizes : numpy.ndarray
            Parent sizes L [um], shape ``(n,)``.
        orders : numpy.ndarray
            Moment orders k, shape ``(m,)``.

        Returns
        -------
        numpy.ndarray
            b_k(L) [um**k], shape ``(n, m)``.
        """
        orders = np.asarray(orders, dtype=float)
        c = self.shape_parameter
        share = (3 * c / (orders + 3) + (1 - c / 2) * (72 / (orders + 9) - 72 / (orders + 6)
                                                        + 18 / (orders + 3)))  # [-]
        return share[None, :] * _size_powers(parent_sizes, orders)


class UniformFragmentation:
    """Fragments spread evenly in volume below the parent's size.

    b(L|lambda) = 6 L**2/lambda**3 on 0 < L < lambda, two fragments on
    average, so b_k = 6/(k+3) lambda**k.
    """

    def moments(self, parent_sizes, orders):
        """Moments of the fragments of each parent.

        Parameters
        ----------
        parent_sizes : numpy.ndarray
            Parent sizes L [um], shape ``(n,)``.
        orders : numpy.ndarray
            Moment orders k, shape ``(m,)``.

        Returns
        -------
        numpy.ndarray
            b_k(L) [um**k], shape ``(n, m)``.
        """
        share = 6 / (np.asarray(orders, dtype=float) + 3)  # [-]
        return share[None, :] * _size_powers(parent_sizes, orders)
