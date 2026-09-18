"""Fail-closed boundary candidates for inventory-constrained reconciliation.

The restricted solve supplies a candidate, not a modified biological model.
Acceptance uses the original constraints and a rational Lagrangian lower bound
on the original problem's necessary linear relaxation.
"""

from fractions import Fraction

import numpy as np
from scipy.linalg import null_space, qr
from scipy.optimize import LinearConstraint, linprog, minimize, nnls


def _affine_basis(matrix, rhs):
    """Eliminate equalities exactly for the supplied binary floating coefficients."""
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
                rows[i] = [a - factor*b for a, b in zip(rows[i], rows[rank])]
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


def _objective_gap(matrix, lower, upper, terms, inequality, capacity, flux):
    """Bound original objective error with exact arithmetic and feasible dual signs."""
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
    F = lambda x: Fraction(float(x))
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
        + sum(F(equality_dual[i])*F(matrix[i, j]) for i in range(len(matrix)))
        - sum(F(multipliers[i])*F(C[i, j]) for i in range(len(C)) if multipliers[i])
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
                 for j in range(count))) for i in range(len(matrix)))
    defect += sum(F(multipliers[i])*max(Fraction(0),
                  -sum(F(C[i,j])*F(flux[j]) for j in range(count))-F(d[i]))
                  for i in range(len(C)) if multipliers[i])
    return float(gap), float(defect), float(objective)


def boundary_candidate(matrix, lower, upper, terms, availability, tolerance):
    """Return a qualified boundary candidate or refuse numerical recovery."""
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
        raw = np.array(basis, dtype=float).reshape(len(lower), -1)
        transform = scale[:, None] * qr(raw / scale[:, None], mode='economic')[0] if raw.shape[1] else raw
        zero = np.array([not any(row) for row in basis])
        transform[zero] = 0
        exposure, _ = availability._exposure(origin)
        if not np.isfinite(exposure):
            return None
        C0 = np.vstack([identity, -identity, exposure * availability.matrix / availability.scale[:, None]])
        d0 = np.r_[-lower, upper, (availability.amounts + availability.other) / availability.scale]
        C, d = C0 @ transform, C0 @ origin + d0
        # Only algebraically constant directions are removed. Never normalize
        # a roundoff remnant of a constraint that is exactly constant.
        F = lambda x: Fraction(float(x))
        keep = np.array([any(sum(F(row[i])*basis[i][j] for i in range(len(lower)))
                        for j in range(raw.shape[1]))
                        for row in np.vstack([identity, -identity, availability.matrix])])
        if exposure == 0:
            keep[2*len(lower):] = False
        if np.any(d[~keep] < -tolerance):
            return None
        C, d = C[keep], d[keep]
        norms = np.max(abs(C), axis=1) if C.shape[1] else np.empty(0)
        if np.any(norms == 0):
            return None
        C, d = C / norms[:, None], d / norms
        from .reconciled import _reconciliation_objective, _reconciliation_gradient
        if transform.shape[1]:
            result = minimize(lambda z: _reconciliation_objective(origin + transform @ z, terms),
                np.linalg.lstsq(transform, seed.x - origin, rcond=None)[0],
                jac=lambda z: transform.T @ _reconciliation_gradient(origin + transform @ z, terms),
                constraints=[LinearConstraint(C, -d, np.inf)], method='SLSQP',
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
            active = C @ result.x + d < 1e-7
            rows = C[active]
            hessian = 2 * transform.T @ (terms.diagonal[:, None] * transform)
            kkt = np.block([[hessian, rows.T],
                            [rows, np.zeros((len(rows), len(rows)))]])
            rhs = np.r_[-2 * transform.T @ (terms.diagonal * origin + terms.linear), -d[active]]
            polished = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:transform.shape[1]]
            flux = origin + transform @ polished
            gap, defect, value = _objective_gap(matrix, lower, upper, terms, inequality, capacity, flux)
            limit = tolerance * max(1., abs(value))
        if not np.isfinite([gap, defect, value]).all() or gap < -defect or gap + defect > limit:
            return None
        return flux, value, gap, defect
    except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError):
        return None
