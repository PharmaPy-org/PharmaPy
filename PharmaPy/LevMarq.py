#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 20:23:12 2019

@author: dcasasor
"""

from numpy import ones_like, inner, diag, asarray, maximum, full, nan
from numpy.linalg import LinAlgError, solve, norm, inv


lm_header = [
    '{:<40}'.format('-'*60),
    "{:<7} {:<10} {:<10} {:<10} {:<10}".format('eval', 'fun_val', '||step||', 'gradient', 'dampening_factor'),
    '{:<40}'.format('-'*60)]


def levenberg_marquardt(x, func, deriv, fletcher_modif=False, max_fun_eval=100,
                        eps_1=1e-8, eps_2=1e-8, tol_fun=1e-12,
                        full_output=False, mu=None, d_diag=None,
                        args=(), verbose=False):
    """Minimize a sum of squared residuals with Levenberg-Marquardt.

    Each trial solves the damped Gauss-Newton system
    ``(A + mu * D.T @ D) @ step = -b`` with ``A = J @ J.T`` and
    ``b = J @ r`` at the accepted point, where ``r = func(x, *args)`` and
    ``J = deriv(x, *args)``. A trial ``x + step`` is accepted when it
    lowers ``norm(r)**2``, i.e. when the gain ratio ``rho``, the actual
    over the predicted decrease, is positive. After an accepted trial
    ``mu`` is multiplied by ``max(1/3, 1 - (beta - 1) * (2*rho - 1)**3)``,
    where ``beta`` is 2 unless the preceding trials were rejected (then
    the current ``nu``), and ``nu`` is reset to 2: ``mu`` decreases, by at
    most a factor of 3, when ``rho > 0.5``, and increases when
    ``0 < rho < 0.5``. A rejected trial is discarded, ``mu`` is multiplied
    by ``nu`` and ``nu`` is doubled.

    Parameters
    ----------
    x : array_like
        Initial decision variables, shape ``(n_x,)``, in parameter units.
    func : callable
        ``func(x, *args)`` returning the residual vector ``r``, shape
        ``(n_data,)``, in residual units (``[-]`` for whitened residuals).
        Either sign convention (model minus data or data minus model)
        works, provided ``deriv`` returns the Jacobian of ``func`` itself:
        the step uses ``b = J @ r``.
    deriv : callable
        ``deriv(x, *args)`` returning the Jacobian of ``func`` with respect
        to ``x``, shape ``(n_x, n_data)`` (one row per decision variable),
        in residual units divided by parameter units.
    fletcher_modif : bool, optional
        If True, use Jacobian-based scaling: after each accepted step, every
        entry of ``D`` is replaced by its parameter's Jacobian row norm
        ``norm(J, axis=1)`` when that is larger. Unless ``d_diag`` is
        given, every entry starts at the largest row norm
        ``max(norm(J, axis=1))`` of the initial point, so ``D`` carries
        Jacobian units. The default is False, which keeps ``D`` at its
        initial value: ``d_diag``, or the identity (unit scaling).
    max_fun_eval : int, optional
        Maximum number of trial evaluations of ``func`` and ``deriv`` [-],
        excluding the evaluation at the initial point. The default is 100.
    eps_1 : float, optional
        Gradient threshold [residual units**2 / parameter units]: stop
        after an accepted step when ``norm(b) < eps_1``. The default is
        1e-8.
    eps_2 : float, optional
        Relative step threshold [-]: stop before a trial when
        ``norm(step) < eps_2 * norm(x)``. The default is 1e-8.
    tol_fun : float, optional
        Residual threshold [residual units]: stop when the residual 2-norm
        ``norm(r)`` at the accepted point is below ``tol_fun``. The default
        is 1e-12.
    full_output : bool, optional
        If True, also return the covariance estimate and a dictionary with
        details about the optimization. The default is False.
    mu : float, optional
        Initial damping factor. ``mu * D.T @ D`` must carry the units of
        ``A``, [residual units**2 / parameter units**2]; with unit scaling
        ``mu`` has those units. The default is None, which uses
        ``1e-2 * max(diag(A))`` at the initial point.
    d_diag : array_like, optional
        Initial diagonal of the scaling matrix ``D``, shape ``(n_x,)``,
        replacing the initial value that ``fletcher_modif`` selects; with
        ``fletcher_modif=True`` its entries are still raised to the
        Jacobian row norms after accepted steps. ``mu * d_diag**2`` must
        carry the units of ``A``. The default is None.
    args : tuple, optional
        Additional positional arguments passed to ``func`` and ``deriv``.
        The default is ().
    verbose : bool, optional
        If True, print a table with the evaluation count, ``norm(r)**2`` of
        the trial point, the step norm, the gradient norm ``norm(b)`` and
        ``mu`` as the optimization proceeds. The default is False.

    Returns
    -------
    x : numpy.ndarray
        Accepted decision variables, shape ``(n_x,)``, in parameter units.
        Returned alone when ``full_output`` is False.
    covar_x : numpy.ndarray
        ``inv(A)`` at the accepted point, shape ``(n_x, n_x)``, entry
        ``(i, j)`` in [parameter i units * parameter j units / residual
        units**2]: the parameter covariance when the residuals are whitened
        [-]. Filled with NaN when ``A`` is exactly singular (``inv`` raises
        ``LinAlgError``), for example when a decision variable does not
        affect any residual; the solve itself is still returned. Returned
        only when ``full_output`` is True.
    output : dict
        Returned only when ``full_output`` is True. ``'x'``, ``'fun'`` and
        ``'jac'`` are the accepted point, its residuals ``r`` and Jacobian
        ``J`` (shapes and units as above). ``'num_iter'`` [-] counts
        accepted steps and ``'num_fun_eval'`` [-] trial evaluations.
        ``'lm_params'`` holds the final ``'d_diag'`` (shape ``(n_x,)``) and
        ``'mu'``. When ``max_fun_eval > 0``, ``'stop_criterion'`` (str)
        states why the loop ended and ``'norm_step'`` is the norm of the
        last solved step [parameter units], which may be a rejected trial
        or the small step that stopped the loop.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the damped system of a trial step is singular. A singular ``A``
        at the accepted point no longer raises (#532); see ``covar_x``.

    Notes
    -----
    The stopping criteria are checked in this order: a small step before
    each trial, then, after an accepted trial, the gradient (``eps_1``)
    followed by the residual norm (``tol_fun``). The residual criterion is
    evaluated on the accepted point, so ``output['fun']`` always belongs to
    the returned ``x``. Because the denominator of ``rho`` is positive for a
    nonzero step and a positive ``mu``, a trial is rejected only when it
    does not lower
    ``norm(r)``; a rejected trial therefore stops on ``tol_fun`` only when
    the accepted point, which can then only be the initial one, already
    satisfies it. Otherwise the loop ends with ``'Maximum iterations
    exceeded'`` once ``max_fun_eval`` trials have been evaluated.

    With decision variables of different units the norms above mix units.
    The default thresholds and ``max_fun_eval`` are long-standing package
    defaults without a recorded derivation; adapt them to the scales of the
    residuals and parameters.

    ``covar_x`` is the plain inverse, without an identifiability test:
    ``inv`` raises only for an exactly singular ``A``. A numerically
    singular ``A`` gives finite entries without correct digits, possibly
    negative variances, and a non-finite Jacobian or an ``A`` close to the
    underflow limit gives non-finite entries without an error.
    ``ParameterEstimation.optimize_fn`` does not use ``covar_x``; its
    covariance comes from ``ParameterEstimation.get_covariance``, which
    tests ``A`` and returns NaN with a warning when it is undefined.
    """

    nu = 2
    beta = nu
    x = asarray(x)

    num_iter = 0

    fun = func(x, *args)
    jac = deriv(x, *args)

    a_matrix = inner(jac, jac)  # Hessian approximation
    b_vector = inner(jac, fun)  # gradient

    if d_diag is None:
        if fletcher_modif:
            d_scaling = norm(jac, axis=1)
            d_scaling = max(d_scaling) * ones_like(x)
        else:
            d_scaling = ones_like(x)
    else:
        d_scaling = d_diag

    if mu is None:
        mu = 1e-2 * max(diag(a_matrix))  # after Nielsen (1999)

    num_feval = 0

    if verbose:
        print('\n'.join(lm_header))
        print("{:<7} {:<10.3e} {:<10} {:<10.3e} {:<10.3e}".format(
            num_feval, norm(fun)**2, '---', norm(b_vector), mu))

    while num_feval < max_fun_eval:
        d_diag = diag(d_scaling)
        lm_step = solve(a_matrix + mu * inner(d_diag.T, d_diag), -b_vector)

        if norm(lm_step) < eps_2 * norm(x):
            reason = 'Small step'
            break

        x_new = x + lm_step
        fun_new = func(x_new, *args)
        jac_new = deriv(x_new, *args)

        num_feval += 1

        sq_old = norm(fun)**2
        sq_new = norm(fun_new)**2
        rho = (sq_old - sq_new) / (inner(lm_step, mu * lm_step - b_vector))

        if rho > 0:  # update x if function is decreased
            x = x_new
            fun = fun_new
            jac = jac_new

            a_matrix = inner(jac, jac)  # Hessian approximation
            b_vector = inner(jac, fun)

            if fletcher_modif:
                d_scaling = maximum(norm(jac, axis=1), d_scaling)

            num_iter += 1

            if norm(b_vector) < eps_1:
                reason = 'Small gradient'
                break

            mu = mu * max(1/3, 1 - (beta - 1) * (2*rho - 1)**3)  # After Nielsen
            nu = 2
        else:
            mu = mu * nu  # decrease step size
            nu = 2 * nu

        beta = nu

        if verbose:
            if num_feval % 50 == 0:
                print('\n'.join(lm_header))

            print("{:<7} {:<10.3e} {:<10.3e} {:<10.3e} {:<10.3e}".format(
                num_feval, sq_new, norm(lm_step), norm(b_vector), mu))

        # x, fun and jac are the accepted point: the residuals of a rejected
        # trial must not end the loop at a point it did not move to.
        if norm(fun) < tol_fun:
            reason = 'Value of the objective function lower than tolerance'
            break

    else:
        reason = 'Maximum iterations exceeded'

    try:
        # [x_i units * x_j units / residual units**2]
        covar_x = inv(a_matrix)
    except LinAlgError:
        # A singular A leaves the covariance undefined, not the solve.
        covar_x = full(a_matrix.shape, nan)

    if verbose:
        print('{:<40}'.format('-'*60))
        print()

    if full_output:
        lm_par = {'d_diag': d_scaling, 'mu': mu}
        if max_fun_eval == 0:
            output_dict = {'x': x, 'fun': fun, 'jac': jac,
                           'num_iter': num_iter, 'num_fun_eval': num_feval,
                           'lm_params': lm_par}
        else:
            output_dict = {'x': x, 'fun': fun, 'jac': jac,
                           'norm_step': norm(lm_step),
                           'stop_criterion': reason, 'num_iter': num_iter,
                           'num_fun_eval': num_feval, 'lm_params': lm_par}

        return x, covar_x, output_dict
    else:
        return x
