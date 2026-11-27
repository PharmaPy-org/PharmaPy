"""Moment inversion for QMOM: Gauss rules from moments of known populations.

Expected nodes and weights are the sizes and counts the moments are built
from, or NumPy's independent Gauss-Laguerre rule; nothing is integrated.
Sizes are in um and counts in #/m3 of slurry, as the crystallizers carry them.
"""

from math import factorial

import numpy as np
import pytest

from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.MomentInversion import invert_moments

pytestmark = pytest.mark.unit

# Rounding through Wheeler's recursion and the eigensolver: the largest
# relative error over these fixtures was 1.0e-12, on the few nuclei at zero
# size beside many seeds, and at most 3.6e-15 elsewhere (29 Oct 2026). This
# keeps a margin of 100 for other platforms' LAPACK.
RECOVERY_RTOL = 1e-10  # [-]

# Assimulo CVode's default relative tolerance, which PharmaPy's integrator
# backends keep: the size of the error an integrated moment can carry.
CVODE_DEFAULT_RTOL = 1e-6  # [-]

SEEDS = ((120.0,), (4e9,))  # [um], [#/m3]
FINES_AND_SEEDS = ((150.0, 8.0), (2e9, 3e11))  # [um], [#/m3], not in size order
THREE_SIZES = ((60.0, 310.0, 5.0), (4e9, 3e7, 1e12))  # [um], [#/m3], not in size order
# Asked for three nodes, rounding alone gives this pair a spurious third.
DUST_AND_SEEDS = ((2.0, 120.0), (2e9, 4e9))  # [um], [#/m3]

# Steady MSMPR crystallizer: n(L) = n0 exp(-L / (G tau)) (Randolph and
# Larson, Theory of Particulate Processes, 1988), so m_k = n0 k! (G tau)**(k+1).
MSMPR_NUCLEI_DENSITY = 1e10  # n0 [#/(um m3)]
MSMPR_SIZE_SCALE = 85.0  # G tau [um]


def population_moments(sizes, counts, num_mom):
    """Moments of a discrete population.

    Parameters
    ----------
    sizes : tuple of float
        Crystal sizes [um].
    counts : tuple of float
        Crystals at each size [#/m3].
    num_mom : int
        Number of moments.

    Returns
    -------
    numpy.ndarray
        m_0..m_{num_mom-1} [um**k/m3], shape ``(num_mom,)``.
    """
    sizes = np.asarray(sizes)
    counts = np.asarray(counts)
    return np.array([np.sum(counts * sizes ** k) for k in range(num_mom)])


def assert_reproduces(rule, moments, num_mom):
    """Check that the rule gives back the leading moments.

    Parameters
    ----------
    rule : QuadratureRule
        Nodes [um] and weights [#/m3].
    moments : numpy.ndarray
        Expected moments [um**k/m3].
    num_mom : int
        How many leading moments the rule must reproduce.
    """
    quadrature = [np.sum(rule.weights * rule.nodes ** k) for k in range(num_mom)]
    np.testing.assert_allclose(quadrature, moments[:num_mom], rtol=RECOVERY_RTOL)


@pytest.mark.parametrize('population', [SEEDS, FINES_AND_SEEDS, THREE_SIZES],
                         ids=['seeds', 'fines_and_seeds', 'three_sizes'])
def test_recovers_the_sizes_and_counts_of_a_discrete_population(population):
    sizes, counts = population
    num_nodes = len(sizes)
    moments = population_moments(sizes, counts, 2 * num_nodes)

    rule = invert_moments(moments, num_nodes)

    order = np.argsort(sizes)
    np.testing.assert_allclose(rule.nodes, np.asarray(sizes)[order], rtol=RECOVERY_RTOL)
    np.testing.assert_allclose(rule.weights, np.asarray(counts)[order], rtol=RECOVERY_RTOL)
    assert_reproduces(rule, moments, 2 * num_nodes)


def test_msmpr_distribution_gives_the_gauss_laguerre_rule():
    num_nodes = 3
    moments = np.array([MSMPR_NUCLEI_DENSITY * factorial(k) * MSMPR_SIZE_SCALE ** (k + 1)
                        for k in range(2 * num_nodes)])  # [um**k/m3]
    laguerre_nodes, laguerre_weights = np.polynomial.laguerre.laggauss(num_nodes)

    rule = invert_moments(moments, num_nodes)

    np.testing.assert_allclose(rule.nodes, MSMPR_SIZE_SCALE * laguerre_nodes,
                               rtol=RECOVERY_RTOL)
    np.testing.assert_allclose(rule.weights, moments[0] * laguerre_weights,
                               rtol=RECOVERY_RTOL)


def test_fewer_nodes_than_moments_allow_reproduce_the_leading_moments():
    moments = population_moments(*THREE_SIZES, 6)

    rule = invert_moments(moments, 2)

    assert len(rule.nodes) == 2
    assert_reproduces(rule, moments, 4)


@pytest.mark.parametrize('population', [SEEDS, DUST_AND_SEEDS],
                         ids=['seeds', 'dust_and_seeds'])
def test_a_population_with_fewer_sizes_than_nodes_keeps_only_its_sizes(population):
    sizes, counts = population
    moments = population_moments(sizes, counts, 6)

    rule = invert_moments(moments, 3)

    np.testing.assert_allclose(rule.nodes, sorted(sizes), rtol=RECOVERY_RTOL)
    np.testing.assert_allclose(rule.weights, np.asarray(counts)[np.argsort(sizes)],
                               rtol=RECOVERY_RTOL)


def test_nuclei_born_at_zero_size_give_one_node_at_zero():
    nuclei = 5e8  # [#/m3], before any growth: every higher moment is zero
    moments = np.array([nuclei, 0.0, 0.0, 0.0, 0.0, 0.0])

    rule = invert_moments(moments, 3)

    np.testing.assert_array_equal(rule.nodes, [0.0])
    np.testing.assert_allclose(rule.weights, [nuclei], rtol=RECOVERY_RTOL)


def test_nuclei_still_at_zero_size_beside_seeds_keep_their_node():
    # Wheeler's cancellation puts the zero-size node at about -4e-11 um here.
    sizes, counts = (0.0, 40.0), (3e7, 1e12)  # [um], [#/m3]
    moments = population_moments(sizes, counts, 4)

    rule = invert_moments(moments, 2)

    assert np.all(rule.nodes >= 0)
    np.testing.assert_allclose(rule.nodes, sizes, rtol=RECOVERY_RTOL,
                               atol=RECOVERY_RTOL * sizes[-1])
    np.testing.assert_allclose(rule.weights, counts, rtol=RECOVERY_RTOL)


def test_nuclei_beside_seeds_keep_their_node_under_integration_error():
    # A first moment low by the integrator's tolerance puts the nuclei node
    # at about -0.02 um; dropping it would merge nuclei and seeds into one.
    sizes, counts = (0.0, 100.0), (5e7, 1e10)  # [um], [#/m3]
    moments = population_moments(sizes, counts, 4)
    moments[1] *= 1 - CVODE_DEFAULT_RTOL

    rule = invert_moments(moments, 2)

    assert rule.nodes[0] == 0.0
    np.testing.assert_allclose(rule.nodes[1], sizes[1], rtol=CVODE_DEFAULT_RTOL)
    np.testing.assert_allclose(rule.weights, counts, rtol=CVODE_DEFAULT_RTOL)


@pytest.mark.parametrize('moments', [np.zeros(6), np.array([-1e3, 0.0, 0.0, 0.0, 0.0, 0.0])],
                         ids=['empty', 'negative_count'])
def test_no_crystals_give_no_nodes(moments):
    rule = invert_moments(moments, 3)

    assert rule.nodes.shape == (0,)
    assert rule.weights.shape == (0,)


def test_moments_no_population_can_have_stop_at_the_last_realizable_node():
    # m_2 below m_1**2 / m_0 would need a negative variance.
    count, mean_size = 1e9, 100.0  # [#/m3], [um]
    moments = np.array([count, count * mean_size, 0.5 * count * mean_size ** 2, 1e15])

    rule = invert_moments(moments, 2)

    np.testing.assert_allclose(rule.nodes, [mean_size], rtol=RECOVERY_RTOL)
    np.testing.assert_allclose(rule.weights, [count], rtol=RECOVERY_RTOL)


def test_a_node_that_needs_a_negative_size_moves_to_zero_with_its_count():
    # Realizable only with a crystal at -5 um, read as one at zero size.
    sizes, counts = (-5.0, 20.0), (1e9, 3e9)  # [um], [#/m3]
    moments = population_moments(sizes, counts, 4)

    rule = invert_moments(moments, 2)

    assert rule.nodes[0] == 0.0
    np.testing.assert_allclose(rule.nodes[1], sizes[1], rtol=RECOVERY_RTOL)
    np.testing.assert_allclose(rule.weights, counts, rtol=RECOVERY_RTOL)


def test_nuclei_with_a_first_moment_just_below_zero_give_one_node_at_zero():
    # Every crystal is still at zero size, and integration error has left
    # m_1 below zero by less than CVode's default absolute tolerance, 1e-6.
    nuclei = 5e8  # [#/m3]
    moments = np.array([nuclei, -1e-7, 0.0, 0.0])  # [um**k/m3]

    rule = invert_moments(moments, 2)

    np.testing.assert_array_equal(rule.nodes, [0.0])
    np.testing.assert_allclose(rule.weights, [nuclei], rtol=RECOVERY_RTOL)


@pytest.mark.parametrize('moments, num_nodes, rtol, message', [
    (np.ones(6), 0, 1e-10, 'num_nodes must be a positive integer'),
    (np.ones(6), 2.0, 1e-10, 'num_nodes must be a positive integer'),
    (np.ones(6), True, 1e-10, 'num_nodes must be a positive integer'),
    (np.ones(5), 3, 1e-10, '3 nodes need a 1-D array of at least 6 moments'),
    (np.ones((6, 2)), 3, 1e-10, '3 nodes need a 1-D array of at least 6 moments'),
    (np.array([1.0, np.nan, 1.0, 1.0]), 2, 1e-10, 'Moments must be finite'),
    (np.array([1.0, 1.0, np.inf, 1.0]), 2, 1e-10, 'Moments must be finite'),
    (np.ones(4), 2, -1e-10, 'rtol must be a finite, non-negative number'),
    (np.ones(4), 2, np.nan, 'rtol must be a finite, non-negative number'),
], ids=['zero_nodes', 'float_nodes', 'bool_nodes', 'too_few_moments', 'two_dimensional',
        'nan_moment', 'infinite_moment', 'negative_rtol', 'nan_rtol'])
def test_invalid_arguments_are_refused(moments, num_nodes, rtol, message):
    with pytest.raises(PharmaPyValueError, match=message):
        invert_moments(moments, num_nodes, rtol)
