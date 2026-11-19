"""Gauss quadrature of a size distribution from its leading moments, for QMOM."""

from typing import NamedTuple

import numpy as np
from scipy.linalg import eigh_tridiagonal

from PharmaPy.Errors import PharmaPyValueError

# A node is kept only while it explains more of the next even moment than
# rounding can: sigma_kk > DEGENERACY_RTOL * |m_2k| (see invert_moments). On
# moment sets holding fewer points than nodes asked for, rounding left at
# most 5.4e-12 over 150,000 random 1- to 3-point sets (sizes 0.1 to 1585
# um, weights 1e6 to 1e14 #/m3, 29 Oct 2026), and with no cut-off 8.8 % of
# them gained a spurious node. This keeps a margin of about twenty. Design
# assumption. It merges a population into one node only when its sizes
# spread by less than a coefficient of variation of 1e-5.
DEGENERACY_RTOL = 1e-10  # [-]


class QuadratureRule(NamedTuple):
    """Nodes and weights of a Gauss quadrature of a size distribution.

    Attributes
    ----------
    nodes : numpy.ndarray
        Sizes L_i in non-decreasing order [x], shape ``(n,)``, all
        non-negative.
    weights : numpy.ndarray
        Positive weights w_i on the basis of the zeroth moment, shape
        ``(n,)``. ``sum(w_i * L_i**k)`` reproduces moment k for k < 2n,
        except for a node moved to zero (see ``invert_moments``).
    """

    nodes: np.ndarray
    weights: np.ndarray


def invert_moments(moments, num_nodes: int,
                   rtol: float = DEGENERACY_RTOL) -> QuadratureRule:
    """Gauss quadrature reproducing the leading moments of a distribution.

    Parameters
    ----------
    moments : array-like
        Moments m_k = integral of L**k n(L) dL, k = 0, 1, ..., [x**k basis],
        shape ``(num_mom,)`` with ``num_mom >= 2 * num_nodes``. Only the
        first ``2 * num_nodes`` are used. For crystals: [um**k/m3 slurry].
    num_nodes : int
        Largest number of nodes N to return, at least 1.
    rtol : float, optional
        Relative tolerance [-] below which the moments are taken to hold no
        further node; see Notes. Defaults to ``DEGENERACY_RTOL``.

    Returns
    -------
    QuadratureRule
        The rule of the n <= N nodes the moments hold, at non-negative
        sizes [x], with positive weights [basis] (for crystals: um and
        #/m3 slurry). ``n = 0`` (empty arrays) when m_0 <= 0.

    Raises
    ------
    PharmaPyValueError
        If ``num_nodes`` is not a positive integer, ``moments`` is not a
        1-D finite array with at least ``2 * num_nodes`` entries, or
        ``rtol`` is negative or not finite.

    Notes
    -----
    Wheeler's (1974, Rocky Mountain J. Math. 4:287-296) recursion turns the
    moments into the recurrence coefficients a_k, b_k of the polynomials
    orthogonal under n(L); John and Thein (2012, Chem. Eng. Sci.
    75:327-333) found it more robust than the product-difference algorithm
    of the original QMOM. The nodes are the eigenvalues of the Jacobi
    matrix (diagonal a_k, off-diagonal sqrt(b_k)) and w_i = m_0 v_0i**2,
    with v_0i the first component of each eigenvector (Golub and Welsch,
    1969, Math. Comp. 23:221-230).

    The rule has fewer than N nodes when the moments support fewer. Wheeler's
    sigma_kk is the squared norm of the k-th orthogonal polynomial, which is
    what the k-node rule misses on m_2k: the rule integrates
    L**2k - P_k(L)**2 exactly and P_k vanishes at its nodes. So the
    recursion stops at k nodes once sigma_kk <= rtol * |m_2k|, both when
    the population holds only k sizes and when the moments are not
    realizable (sigma_kk < 0).

    A node below zero is moved to zero with its weight kept, which changes
    moment k by w_i * |L_i|**k. Crystals at zero size, such as nuclei born
    at rad = 0, sit on the edge of the support, and rounding or integration
    error puts their node slightly below zero. Dropping to a smaller rule
    instead collapsed QMOM solves for good: the smaller closure moved the
    higher moments so that no larger rule was non-negative again, and with
    nucleation at zero size m_5 ended up to 86 % off under CVode at its
    default tolerances (probe, 3 Nov 2026). A negative m_1 thus gives one
    node at zero holding m_0.
    """
    moments = _validated_moments(moments, num_nodes, rtol)[:2 * num_nodes]  # [x**k basis]

    if not moments[0] > 0:
        return QuadratureRule(np.empty(0), np.empty(0))

    diagonal, off_diagonal_sq = _recurrence_coefficients(moments, rtol)  # [x], [x**2]
    nodes, weights = _gauss_rule(moments[0], diagonal, off_diagonal_sq)  # [x], [basis]

    return QuadratureRule(np.maximum(nodes, 0.0), weights)


def _validated_moments(moments, num_nodes, rtol):
    """Check the arguments of ``invert_moments``.

    Parameters
    ----------
    moments : array-like
        Moments m_k [x**k basis].
    num_nodes : int
        Largest number of nodes.
    rtol : float
        Degeneracy tolerance [-].

    Returns
    -------
    numpy.ndarray
        The moments as a float array [x**k basis], shape ``(num_mom,)``.

    Raises
    ------
    PharmaPyValueError
        On any argument ``invert_moments`` documents as invalid.
    """
    if (isinstance(num_nodes, (bool, np.bool_))
            or not isinstance(num_nodes, (int, np.integer)) or num_nodes < 1):
        raise PharmaPyValueError(
            'num_nodes must be a positive integer, got %r.' % (num_nodes,))

    if not np.isfinite(rtol) or rtol < 0:
        raise PharmaPyValueError(
            'rtol must be a finite, non-negative number, got %r.' % (rtol,))

    moments = np.asarray(moments, dtype=float)
    if moments.ndim != 1 or len(moments) < 2 * num_nodes:
        raise PharmaPyValueError(
            '%d nodes need a 1-D array of at least %d moments, got shape %s.'
            % (num_nodes, 2 * num_nodes, moments.shape))

    if not np.all(np.isfinite(moments)):
        raise PharmaPyValueError(
            'Moments must be finite, got %s.' % moments.tolist())

    return moments


def _recurrence_coefficients(moments, rtol):
    """Wheeler's recursion, stopped where the moments hold no further node.

    Parameters
    ----------
    moments : numpy.ndarray
        Moments m_0..m_{2N-1} [x**k basis], shape ``(2N,)``, with m_0 > 0.
    rtol : float
        Degeneracy tolerance [-]; see ``invert_moments``.

    Returns
    -------
    diagonal : numpy.ndarray
        a_0..a_{n-1} [x], shape ``(n,)``, 1 <= n <= N.
    off_diagonal_sq : numpy.ndarray
        b_1..b_{n-1} [x**2], shape ``(n - 1,)``, all positive.
    """
    num_mom = len(moments)
    num_nodes = num_mom // 2

    diagonal = np.zeros(num_nodes)  # a_k [x]
    off_diagonal_sq = np.zeros(num_nodes)  # b_0 = m_0 [basis], then b_k [x**2]
    diagonal[0] = moments[1] / moments[0]
    off_diagonal_sq[0] = moments[0]

    # sigma_{k,l} for rows k-2 and k-1, indexed by l: row -1 is zero, row 0
    # holds the moments themselves [x**(l+k) basis].
    sigma_older = np.zeros(num_mom)
    sigma_old = moments.copy()

    for k in range(1, num_nodes):
        span = slice(k, num_mom - k)
        sigma = np.zeros(num_mom)
        sigma[span] = (sigma_old[k + 1:num_mom - k + 1]
                       - diagonal[k - 1] * sigma_old[span]
                       - off_diagonal_sq[k - 1] * sigma_older[span])

        if sigma[k] <= rtol * abs(moments[2 * k]):
            return diagonal[:k], off_diagonal_sq[1:k]

        diagonal[k] = sigma[k + 1] / sigma[k] - sigma_old[k] / sigma_old[k - 1]
        off_diagonal_sq[k] = sigma[k] / sigma_old[k - 1]
        sigma_older, sigma_old = sigma_old, sigma

    return diagonal, off_diagonal_sq[1:]


def _gauss_rule(zeroth_moment, diagonal, off_diagonal_sq):
    """Nodes and weights from the Jacobi matrix (Golub-Welsch).

    Parameters
    ----------
    zeroth_moment : float
        m_0 [basis].
    diagonal : numpy.ndarray
        a_0..a_{n-1} [x], shape ``(n,)``.
    off_diagonal_sq : numpy.ndarray
        b_1..b_{n-1} [x**2], shape ``(n - 1,)``, all positive.

    Returns
    -------
    nodes : numpy.ndarray
        Eigenvalues in ascending order [x], shape ``(n,)``.
    weights : numpy.ndarray
        m_0 times the squared first eigenvector components [basis],
        shape ``(n,)``.
    """
    if len(diagonal) == 1:
        return diagonal.copy(), np.array([zeroth_moment])

    nodes, vectors = eigh_tridiagonal(diagonal, np.sqrt(off_diagonal_sq))
    return nodes, zeroth_moment * vectors[0] ** 2
