"""Fail-closed boundary candidates for inventory-constrained reconciliation.

The restricted solve supplies a candidate, not a modified biological model.
Acceptance uses the original constraints and a rational Lagrangian lower bound
on the original problem's necessary linear relaxation.
"""

from functools import partial
from fractions import Fraction

import numpy as np
from scipy.linalg import null_space, qr
from scipy.optimize import LinearConstraint, linprog, minimize, nnls


from .reconciled import _reconciliation_objective, _reconciliation_gradient
# Keep certification arithmetic exact for the supplied floating coefficients.
def _as_fraction(x):
    """Represent a floating coefficient exactly as a rational number.

    Parameters
    ----------
    x : float
        Finite coefficient, retaining the physical unit of the caller's matrix or objective term.

    Returns
    -------
    Fraction
        Exact representation, retaining the input coefficient's physical unit.

    Raises
    ------
    ValueError
        The coefficient is NaN.
    OverflowError
        The coefficient is infinite.
    """
    return Fraction(float(x))


# Adapt the original objective and gradient to candidate coordinates.
def _boundary_objective(z: np.ndarray, *, origin, transform, terms) -> float:
    """Evaluate the original objective in boundary-candidate coordinates.

    Parameters
    ----------
    z : ndarray
        Normalized optimizer coordinates [-], in the unchanged free/tangent-coordinate order.
    origin : ndarray
        Affine origin: full physical fluxes in the boundary recovery, normalized free coordinates [-] in reconciliation.
    transform : ndarray
        Affine basis from optimizer coordinates to the origin's coordinate basis; row/column order is unchanged.
    terms : _QuadraticTerms
        Dimensionless quadratic objective with coefficients on the declared flux basis.

    Returns
    -------
    float
        Original dimensionless objective [-].
    """
    return _reconciliation_objective(origin + transform @ z, terms)


def _boundary_gradient(z: np.ndarray, *, transform, origin, terms) -> np.ndarray:
    """Project the original gradient onto boundary-candidate coordinates.

    Parameters
    ----------
    z : ndarray
        Normalized optimizer coordinates [-], in the unchanged free/tangent-coordinate order.
    transform : ndarray
        Affine basis from optimizer coordinates to the origin's coordinate basis; row/column order is unchanged.
    origin : ndarray
        Affine origin: full physical fluxes in the boundary recovery, normalized free coordinates [-] in reconciliation.
    terms : _QuadraticTerms
        Dimensionless quadratic objective with coefficients on the declared flux basis.

    Returns
    -------
    ndarray
        Gradient with respect to normalized candidate coordinates [-].
    """
    return transform.T @ _reconciliation_gradient(origin + transform @ z, terms)


# Eliminate equality constraints exactly before proposing a candidate.
def _affine_basis(matrix, rhs):
    """Eliminate equalities exactly for the supplied binary floating coefficients.

    Parameters
    ----------
    matrix : ndarray, shape (m, n)
        Equality coefficients; each float is converted exactly to a fraction.
    rhs : ndarray, shape (m,)
        Equality right-hand side.

    Returns
    -------
    tuple of list
        ``(origin, basis)``: one exact solution (length n) and an exact null
        space basis (n rows), as ``Fraction`` values.

    Raises
    ------
    ValueError
        The equalities are inconsistent.
    """
    rows = [[Fraction(float(x)) for x in row] + [Fraction(float(b))]
            for row, b in zip(matrix, rhs)]
    columns = matrix.shape[1]
    pivots = []
    for column in range(columns):
        rank = len(pivots)
        selected = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if selected is None:
            continue
        rows[rank], rows[selected] = rows[selected], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [x / divisor for x in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][column]:
                factor = rows[i][column]
                rows[i] = [a - factor*b if b else a for a, b in zip(rows[i], rows[rank])]
        pivots.append(column)
    if any(not any(row[:-1]) and row[-1] for row in rows):
        raise ValueError('inconsistent candidate equalities')
    free = [i for i in range(columns) if i not in pivots]
    basis = [[Fraction(0) for _ in free] for _ in range(columns)]
    origin = [Fraction(0) for _ in range(columns)]
    for i, pivot in enumerate(pivots):
        origin[pivot] = rows[i][-1]
    for j, column in enumerate(free):
        basis[column][j] = Fraction(1)
        for i, pivot in enumerate(pivots):
            basis[pivot][j] = -rows[i][column]
    return origin, basis


# Certify the original problem independently of optimizer status.
def _objective_gap(matrix, lower, upper, terms, inequality, capacity, flux):
    """Bound original objective error with exact arithmetic and feasible dual signs.

    Parameters
    ----------
    matrix : ndarray
        Stoichiometric matrix.
    lower, upper : ndarray
        Finite flux bounds.
    terms : _QuadraticTerms
        Objective terms.
    inequality, capacity : ndarray
        Linear inventory relaxation ``A v <= b``.
    flux : ndarray
        Candidate fluxes.

    Returns
    -------
    tuple of float
        ``(gap, defect, objective)``: candidate objective minus an exact
        Lagrangian lower bound, the allowance for the candidate's constraint
        residuals, and the candidate objective, all [-].

    Notes
    -----
    Floating linear algebra only proposes multipliers; the lower bound is valid
    for any nonnegative multipliers. Constraints within 1e-7 of activity, and
    rows with norm above 1e-12, are used to propose multipliers only.
    """
    count = len(lower)
    scale = np.ones(count)
    positive = terms.diagonal > 0
    scale[positive] = 1 / np.sqrt(terms.diagonal[positive])
    transform = scale[:, None] * null_space(matrix * scale)
    C = np.vstack([np.eye(count), -np.eye(count), -inequality])
    d = np.r_[-lower, upper, capacity]
    projected = C @ transform
    norms = np.linalg.norm(projected, axis=1)
    keep = norms > 1e-12
    gradient = 2 * (terms.diagonal * flux + terms.linear)
    multipliers = np.zeros(len(d))
    if transform.shape[1] and keep.any():
        active = (C @ flux + d)[keep] / norms[keep] < 1e-7
        if active.any():
            multipliers[np.flatnonzero(keep)[active]] = nnls(
                (projected[keep] / norms[keep, None])[active].T,
                transform.T @ gradient, maxiter=10000)[0] / norms[keep][active]
    equality_dual = np.linalg.lstsq((matrix * scale).T,
        (C.T @ multipliers - gradient) * scale, rcond=None)[0]
    # Floating linear algebra only proposes multipliers. The following bound
    # remains valid for any such multipliers; stationarity need not be exact.
    F = _as_fraction
    quadratic = [Fraction(0) for _ in lower]
    linear = quadratic.copy()
    constant = Fraction(0)
    for i in terms.internal_positions:
        quadratic[i] += 1 / F(terms.internal_scale)**2
    for i, value, scale_i in zip(terms.target_positions, terms.target_values, terms.target_scales):
        weight = 1 / F(scale_i)**2
        quadratic[i] += weight
        linear[i] -= F(value) * weight
        constant += F(value)**2 * weight
    lagrangian_linear = [2*linear[j]
        + sum(F(equality_dual[i])*F(matrix[i, j]) for i in range(len(matrix))
              if equality_dual[i] and matrix[i, j])
        - sum(F(multipliers[i])*F(C[i, j]) for i in range(len(C)) if multipliers[i] and C[i, j])
        for j in range(count)]
    bound = constant - sum(F(multipliers[i])*F(d[i]) for i in range(len(d)))
    # Minimize each separable Lagrangian term on the declared finite box.
    # This also covers directions without quadratic regularization.
    for q, l, lo, hi in zip(quadratic, lagrangian_linear, lower, upper):
        point = min(max(-l/(2*q), F(lo)), F(hi)) if q else F(lo if l >= 0 else hi)
        bound += q*point**2 + l*point
    objective = constant + sum(q*F(x)**2 + 2*l*F(x)
                               for q, l, x in zip(quadratic, linear, flux))
    gap = objective - bound
    # An exact arithmetic allowance for the candidate's floating reconstruction:
    # its constraint residuals can put its objective slightly below the optimum.
    defect = sum(abs(F(equality_dual[i]) * sum(F(matrix[i,j])*F(flux[j])
                 for j in range(count) if matrix[i,j] and flux[j]))
                 for i in range(len(matrix)) if equality_dual[i])
    defect += sum(F(multipliers[i])*max(Fraction(0),
                  -sum(F(C[i,j])*F(flux[j]) for j in range(count) if C[i,j] and flux[j])-F(d[i]))
                  for i in range(len(C)) if multipliers[i])
    return float(gap), float(defect), float(objective)


# Propose a boundary candidate and accept only its original-problem certificate.
def boundary_candidate(matrix, lower, upper, terms, availability, tolerance):
    """Return a qualified boundary candidate or refuse numerical recovery.

    Parameters
    ----------
    matrix : ndarray
        Stoichiometric matrix.
    lower, upper : ndarray
        Flux bounds; recovery requires finite bounds.
    terms : _QuadraticTerms
        Objective terms.
    availability : LinearInventoryAvailability or None
        Inventory constraint; recovery requires one.
    tolerance : float
        Relative acceptance tolerance on the certified gap [-].

    Returns
    -------
    tuple or None
        ``(flux, objective, gap, defect)`` for a certified candidate, or None.
    """
    if availability is None or not np.isfinite(np.r_[lower, upper]).all():
        return None
    inequality, capacity = availability.linear_relaxation(lower, upper)
    if inequality is None:
        inequality, capacity = np.empty((0, len(lower))), np.empty(0)
    growth = availability.growth_index
    objective = np.zeros(len(lower))
    if growth is not None:
        objective[growth] = -1
    seed = linprog(objective, A_eq=matrix, b_eq=np.zeros(len(matrix)),
                   A_ub=inequality if len(capacity) else None,
                   b_ub=capacity if len(capacity) else None,
                   bounds=list(zip(lower, upper)), method='highs')
    if not seed.success:
        return None
    identity = np.eye(len(lower))
    lo_active = (seed.lower.marginals != 0) | (lower == upper)
    hi_active = (seed.upper.marginals != 0) & ~lo_active
    if growth is not None:
        lo_active[growth] = hi_active[growth] = False
    active = (seed.ineqlin.marginals != 0) & (capacity >= 0)
    equations = np.vstack([matrix, identity[lo_active], identity[hi_active], inequality[active]])
    rhs = np.r_[np.zeros(len(matrix)), lower[lo_active], upper[hi_active], np.zeros(sum(active))]
    if growth is not None:
        equations = np.vstack([equations, identity[growth]])
        rhs = np.r_[rhs, lower[growth]]
    try:
        origin_exact, basis = _affine_basis(equations, rhs)
        origin = np.array(origin_exact, dtype=float)
        scale = np.ones(len(lower))
        positive = terms.diagonal > 0
        scale[positive] = 1 / np.sqrt(terms.diagonal[positive])
        raw_basis = np.array(basis, dtype=float).reshape(len(lower), -1)
        if raw_basis.shape[1]:
            scaled_basis = raw_basis / scale[:, None]  # [1/model flux unit]
            orthonormal_basis, _ = qr(scaled_basis, mode='economic')  # [-]
            transform = scale[:, None] * orthonormal_basis  # [model flux unit]
        else:
            transform = raw_basis  # [-], empty basis: the equalities determine all fluxes
        zero = np.array([not any(row) for row in basis])
        transform[zero] = 0
        exposure, _ = availability._exposure(origin)
        if not np.isfinite(exposure):
            return None
        inventory_coefficients = exposure * availability.matrix / availability.scale[:, None]  # [1/model flux unit]
        inventory_margin = (availability.amounts + availability.other) / availability.scale  # [-]
        constraint_matrix = np.vstack([identity, -identity, inventory_coefficients])
        constraint_offsets = np.r_[-lower, upper, inventory_margin]  # [flux units for bounds, - for inventory]
        projected_constraints = constraint_matrix @ transform
        projected_offsets = constraint_matrix @ origin + constraint_offsets
        # Only algebraically constant directions are removed. Never normalize
        # a roundoff remnant of a constraint that is exactly constant.
        original_rows = np.vstack([identity, -identity, availability.matrix])
        keep = np.zeros(len(original_rows), dtype=bool)
        for row_index, row in enumerate(original_rows):
            for direction_index in range(raw_basis.shape[1]):
                coefficient = Fraction(0)  # [row unit], exact projected coefficient
                for flux_index in np.flatnonzero(row):
                    basis_coefficient = basis[flux_index][direction_index]  # [-]
                    if basis_coefficient:
                        coefficient += _as_fraction(row[flux_index]) * basis_coefficient
                if coefficient:
                    keep[row_index] = True
                    break
        if exposure == 0:
            keep[2*len(lower):] = False
        if np.any(projected_offsets[~keep] < -tolerance):
            return None
        projected_constraints, projected_offsets = projected_constraints[keep], projected_offsets[keep]
        norms = np.max(abs(projected_constraints), axis=1) if projected_constraints.shape[1] else np.empty(0)
        if np.any(norms == 0):
            return None
        projected_constraints, projected_offsets = projected_constraints / norms[:, None], projected_offsets / norms
        if transform.shape[1]:
            result = minimize(partial(_boundary_objective, origin=origin, transform=transform, terms=terms),
                np.linalg.lstsq(transform, seed.x - origin, rcond=None)[0],
                jac=partial(_boundary_gradient, transform=transform, origin=origin, terms=terms),
                constraints=[LinearConstraint(projected_constraints, -projected_offsets, np.inf)], method='SLSQP',
                options={'ftol': 1e-10, 'maxiter': 1000})
            if not result.success:
                return None
            flux = origin + transform @ result.x
        else:
            flux = origin
        gap, defect, value = _objective_gap(matrix, lower, upper, terms, inequality, capacity, flux)
        limit = tolerance * max(1., abs(value))
        if transform.shape[1] and (gap < -defect or gap + defect > limit):
            # Refine a nearly active solution through its linear KKT equations.
            # These extra equalities only generate a candidate; the unchanged
            # original-problem certificate decides whether it may be accepted.
            active = projected_constraints @ result.x + projected_offsets < 1e-7
            rows = projected_constraints[active]
            hessian = 2 * transform.T @ (terms.diagonal[:, None] * transform)
            kkt = np.block([[hessian, rows.T],
                            [rows, np.zeros((len(rows), len(rows)))]])
            rhs = np.r_[-2 * transform.T @ (terms.diagonal * origin + terms.linear), -projected_offsets[active]]
            polished = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:transform.shape[1]]
            flux = origin + transform @ polished
            gap, defect, value = _objective_gap(matrix, lower, upper, terms, inequality, capacity, flux)
            limit = tolerance * max(1., abs(value))
        if not np.isfinite([gap, defect, value]).all() or gap < -defect or gap + defect > limit:
            return None
        return flux, value, gap, defect
    except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError):
        return None
