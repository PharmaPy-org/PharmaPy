"""Unit tests for custom-optimizer adapters (#347) and SciPy record/option guards (#534, #536).

Scope:
- ``ParameterEstimation.optimize_fn`` and ``estimate_parameters`` with a custom
  optimizer callable ``optimizer(seed, objective_fun, gradient_fun, **options)``
  (passed either as ``method=<callable>`` or ``method='custom'`` with
  ``optim_options={'optimizer': <callable>}``), including fixed-parameter
  masks, return-type normalization (mapping, ``OptimizeResult``, 2-tuple,
  1-D array), canonical ``info['jac']`` orientation, covariance, confidence
  intervals, and rollback on failure (#347).
- ``CalibrationRecord.from_estimation`` and ``CalibrationRecord.rerun`` for
  ``'scipy'``, ``'least_squares'``, and ``'custom'`` fits with and without
  parameter bounds (#534).
- ``ParameterEstimation.optimize_fn`` and ``estimate_parameters`` rejection of
  reserved ``scipy.optimize.least_squares`` arguments (``fun``, ``x0``,
  ``jac``, ``args``) in ``optim_options`` with a clear ``ValueError`` (#536).
Core lane; under a second.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision
from PharmaPy.StatsModule import StatisticsClass

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# First-order batch decay fixture: c_A(t) = c_A0 * exp(-k * t)
# Governing equation: dc_A/dt = -k * c_A, integrated analytically.
# ---------------------------------------------------------------------------
TIME_GRID_S = np.array([0.0, 1.0, 2.0, 4.0, 6.0], dtype=float)  # [s]
TRUE_CA0 = 2.0  # [mol/L], fixed initial concentration of species A
TRUE_K = 0.35  # [1/s], true first-order decay rate constant
SEED_K = 0.15  # [1/s], initial guess for rate constant k
MEAS_STD = 0.01  # [mol/L], declared measurement standard deviation
# Small deterministic measurement perturbations [mol/L] so residuals are
# non-zero and covariance scaling with MSE is non-degenerate.
MEAS_OFFSETS = np.array([0.002, -0.003, 0.001, -0.002, 0.002])  # [mol/L]

# Bounds on k [1/s] enclosing TRUE_K = 0.35 1/s in their interior.
LOWER_K = 0.05  # [1/s]
UPPER_K = 1.50  # [1/s]

# Convergence and comparison tolerances [-]: Gauss-Newton and SciPy TRF
# solves use 1e-10 step/gradient tolerances, so estimates agree to < 1e-6.
# Scalar BFGS uses double-precision forward differences dx_finitediff = 1e-8
# [1/s] (near sqrt(eps)) with gradient tolerance gtol = 1e-5 [s].
SOLVER_TOL = 1e-10  # [-]
BFGS_FD_STEP = 1e-8  # [1/s]
BFGS_GTOL = 1e-5  # [s]
ESTIMATE_RTOL = 1e-5  # [-]
ESTIMATE_ATOL = 1e-7  # [1/s]
MAX_GN_ITER = 40  # [-], damped Gauss-Newton iteration limit


def first_order_decay(params: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Evaluate first-order batch decay concentration profile ``c_A(t)``.

    Parameters
    ----------
    params : numpy.ndarray
        Model parameter vector ``[c_A0, k]`` with initial concentration
        ``c_A0`` [mol/L] and first-order rate constant ``k`` [1/s], or
        ``[k]`` [1/s] when ``c_A0`` is fixed at ``TRUE_CA0``.
    time_s : numpy.ndarray
        Sampling times [s], shape ``(n_times,)``.

    Returns
    -------
    numpy.ndarray
        Concentration profile ``c_A(t)`` [mol/L], shape ``(n_times, 1)``.
    """
    params_arr = np.asarray(params, dtype=float)
    if params_arr.size == 1:
        ca0 = TRUE_CA0  # [mol/L]
        rate_k = float(params_arr[0])  # [1/s]
    else:
        ca0 = float(params_arr[0])  # [mol/L]
        rate_k = float(params_arr[1])  # [1/s]
    conc = ca0 * np.exp(-rate_k * np.asarray(time_s, dtype=float))  # [mol/L]
    return conc[:, np.newaxis]


def make_decay_experiments() -> dict[str, Experiment]:
    """Construct a single-experiment ``Experiment`` mapping for decay fitting.

    Returns
    -------
    dict of str to Experiment
        Experiment mapping ``{'batch_1': Experiment(...)}`` with declared
        units ``'mol/L'``, ``x_units='s'``, and uncertainty ``MEAS_STD``.
    """
    exact_ca = first_order_decay(
        np.array([TRUE_CA0, TRUE_K]), TIME_GRID_S)[:, 0]  # [mol/L]
    observed_ca = exact_ca + MEAS_OFFSETS  # [mol/L]
    measurement = Measurement(
        0,
        TIME_GRID_S,
        observed_ca,
        uncertainty=MEAS_STD,
        units='mol/L',
        basis='molar concentration',
        x_units='s',
    )
    return {'batch_1': Experiment({'c_A': measurement})}


def custom_gauss_newton(
        seed: np.ndarray,
        objective_fun,
        gradient_fun,
        *,
        tol: float = SOLVER_TOL,
        max_iter: int = MAX_GN_ITER,
        bounds=None,
        return_kind: str = 'dict'):
    """Solve nonlinear least squares via Gauss-Newton on residual callbacks.

    Parameters
    ----------
    seed : numpy.ndarray
        Initial variable parameter vector, shape ``(num_params,)``, in model
        parameter units.
    objective_fun : callable
        ``ParameterEstimation.get_objective``; called with ``out_array=True``
        to obtain the weighted residual vector ``r`` [-].
    gradient_fun : callable
        ``ParameterEstimation.get_gradient``; called with ``out_array=True``
        to obtain the weighted residual Jacobian ``J`` of shape
        ``(num_params, n_data)`` [1 / parameter units].
    tol : float, optional
        Step-norm convergence tolerance [-]. The default is ``SOLVER_TOL``.
    max_iter : int, optional
        Maximum Gauss-Newton iterations [-]. The default is ``MAX_GN_ITER``.
    bounds : sequence, optional
        Optional parameter bounds in parameter units.
    return_kind : {'dict', 'optimize_result', 'tuple', 'array'}, optional
        Return format used to exercise ``_unpack_custom_optimizer_result``.

    Returns
    -------
    dict or OptimizeResult or tuple or numpy.ndarray
        Solved parameter vector and optional diagnostics in the requested
        return format.
    """
    x_curr = np.array(seed, dtype=float, copy=True)  # [1/s]
    lo_bound = -np.inf  # [1/s]
    hi_bound = np.inf  # [1/s]
    if bounds is not None:
        if (isinstance(bounds, (list, tuple))
                and len(bounds) == len(x_curr)
                and isinstance(bounds[0], (list, tuple))):
            lo_bound = np.array(
                [-np.inf if b[0] is None else float(b[0]) for b in bounds])
            hi_bound = np.array(
                [np.inf if b[1] is None else float(b[1]) for b in bounds])
        elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            lo_bound = np.asarray(bounds[0], dtype=float)
            hi_bound = np.asarray(bounds[1], dtype=float)

    nfev = 0  # [-]
    njev = 0  # [-]
    nit = 0  # [-]
    for iteration in range(1, max_iter + 1):
        nit = iteration
        resid = objective_fun(x_curr, out_array=True)  # [-]
        nfev += 1
        jac = gradient_fun(x_curr, out_array=True)  # [s], shape (n_par, n_data)
        njev += 1
        # Gauss-Newton normal equations: (J @ J.T) * step = -J @ resid
        hess_approx = jac @ jac.T  # [s**2]
        grad_vec = jac @ resid  # [s]
        step = np.linalg.solve(hess_approx, -grad_vec)  # [1/s]
        x_next = np.clip(x_curr + step, lo_bound, hi_bound)  # [1/s]
        x_curr = x_next
        if float(np.linalg.norm(step)) <= tol:
            break

    # Evaluate a perturbed trial point after convergence to verify that
    # optimize_fn refreshes model buffers at the accepted x_curr (#347).
    _ = objective_fun(x_curr + 0.05, out_array=True)
    nfev += 1

    meta = {
        'x': x_curr.copy(),
        'success': True,
        'status': 1,
        'message': 'Gauss-Newton step norm below tol',
        'nit': nit,
        'nfev': nfev,
        'njev': njev,
        'num_iter': nit,
        'num_fun_eval': nfev,
        'stop_criterion': 'Step norm below tolerance',
    }
    if return_kind == 'dict':
        return meta
    if return_kind == 'optimize_result':
        return OptimizeResult(**meta)
    if return_kind == 'tuple':
        info_only = {k: v for k, v in meta.items() if k != 'x'}
        return x_curr.copy(), info_only
    if return_kind == 'array':
        return x_curr.copy()
    raise ValueError(f"Unsupported return_kind {return_kind!r}")


def scipy_bfgs_scalar_optimizer(
        seed: np.ndarray,
        objective_fun,
        gradient_fun,
        *,
        gtol: float = SOLVER_TOL) -> OptimizeResult:
    """Minimize the scalar least-squares objective with SciPy BFGS.

    Exercises the default ``out_array=False`` scalar-objective and 1-D
    gradient contract of ``objective_fun(x)`` and ``gradient_fun(x)``.

    Parameters
    ----------
    seed : numpy.ndarray
        Initial variable parameter vector [1/s].
    objective_fun : callable
        Scalar objective callback ``1/2 * r @ r`` [-].
    gradient_fun : callable
        Scalar-objective gradient callback ``J @ r`` [s].
    gtol : float, optional
        Gradient norm tolerance [s]. The default is ``SOLVER_TOL``.

    Returns
    -------
    scipy.optimize.OptimizeResult
        BFGS optimization result.
    """
    return minimize(
        objective_fun,
        seed,
        method='BFGS',
        jac=gradient_fun,
        options={'gtol': gtol},
    )


# ---------------------------------------------------------------------------
# Slice 4A (#347): Custom optimizer callable adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'return_kind',
    ['dict', 'optimize_result', 'tuple', 'array'],
)
def test_custom_optimizer_return_kinds_and_fixed_parameter_mask(
        return_kind: str):
    """Verify all custom-optimizer return formats and fixed-parameter masks."""
    experiments = make_decay_experiments()
    # Reference fit with native 'scipy' least_squares and fixed c_A0.
    ref_est = ParameterEstimation(
        first_order_decay,
        [TRUE_CA0, SEED_K],
        experiments,
        optimize_flags=[False, True],
        name_params=['c_A0', 'k'],
    )
    ref_par, ref_cov, ref_info = ref_est.optimize_fn(
        method='scipy', verbose=False)

    est = ParameterEstimation(
        first_order_decay,
        [TRUE_CA0, SEED_K],
        experiments,
        optimize_flags=[False, True],
        name_params=['c_A0', 'k'],
    )
    opt_par, covar, info = est.optimize_fn(
        method=custom_gauss_newton,
        optim_options={'tol': SOLVER_TOL, 'return_kind': return_kind},
        bounds=[(LOWER_K, UPPER_K)],
        verbose=False,
    )

    assert est.opt_method == 'custom'
    assert est._custom_optimizer is custom_gauss_newton
    np.testing.assert_allclose(
        opt_par, ref_par, rtol=ESTIMATE_RTOL, atol=ESTIMATE_ATOL)
    np.testing.assert_allclose(
        covar, ref_cov, rtol=1e-4, atol=1e-7)
    # Canonical Jacobian shape (num_params, n_data) = (1, 5)
    assert info['jac'].shape == (1, len(TIME_GRID_S))
    assert est.sens.shape == (len(TIME_GRID_S), 1)
    np.testing.assert_allclose(
        info['fun'], ref_info['fun'], rtol=1e-4, atol=1e-6)
    # Verify final model buffers match accepted opt_par despite the trailing
    # perturbed evaluation inside custom_gauss_newton.
    np.testing.assert_array_equal(est.params_residuals, opt_par)

    intervals = est.get_confidence_intervals(verbose=False)
    assert 'k' in intervals
    stats = StatisticsClass(est)
    stats_intervals = stats.get_intervals(verbose=False)
    np.testing.assert_allclose(
        intervals['k'], stats_intervals['k'], rtol=1e-12, atol=1e-12)


def test_custom_optimizer_via_method_custom_and_estimate_parameters():
    """Verify ``method='custom'`` with ``optim_options={'optimizer': ...}``."""
    experiments = make_decay_experiments()
    est = ParameterEstimation(
        first_order_decay,
        [SEED_K],
        experiments,
        name_params=['k'],
        dx_finitediff=BFGS_FD_STEP,
    )
    opt_par, covar, info = est.estimate_parameters(
        method='custom',
        optimizer=scipy_bfgs_scalar_optimizer,
        gtol=BFGS_GTOL,
        verbose=False,
    )

    assert est.opt_method == 'custom'
    assert est._custom_optimizer is scipy_bfgs_scalar_optimizer
    assert opt_par.shape == (1,)
    assert covar.shape == (1, 1)
    assert info['jac'].shape == (1, len(TIME_GRID_S))
    assert info['success'] is True
    assert float(opt_par[0]) == pytest.approx(TRUE_K, abs=5e-3)


def test_custom_optimizer_validation_and_rollback_on_failure():
    """Invalid custom-optimizer specifications or results preserve prior fit."""
    experiments = make_decay_experiments()
    est = ParameterEstimation(
        first_order_decay,
        [SEED_K],
        experiments,
        name_params=['k'],
        dx_finitediff=BFGS_FD_STEP,
    )
    lm_par, lm_cov, _ = est.optimize_fn(method='LM', verbose=False)

    # 1. method='custom' without 'optimizer' in optim_options
    with pytest.raises(ValueError, match="requires a callable 'optimizer'"):
        est.optimize_fn(method='custom', verbose=False)
    assert est.opt_method == 'LM'
    np.testing.assert_array_equal(est.params_convg, lm_par)

    # 2. Both method=<callable> and optim_options={'optimizer': ...}
    with pytest.raises(ValueError, match="not both"):
        est.optimize_fn(
            method=custom_gauss_newton,
            optim_options={'optimizer': custom_gauss_newton},
            verbose=False,
        )
    assert est.opt_method == 'LM'

    # 3. Duplicate bounds in bounds= and optim_options
    with pytest.raises(
            ValueError,
            match="Pass bounds either through bounds= or optim_options"):
        est.optimize_fn(
            method=custom_gauss_newton,
            bounds=[(LOWER_K, UPPER_K)],
            optim_options={'bounds': [(LOWER_K, UPPER_K)]},
            verbose=False,
        )

    # 4. Mapping result missing 'x'
    with pytest.raises(ValueError, match="must contain key 'x'"):
        est.optimize_fn(
            method=lambda s, f, g: {'status': 0},
            verbose=False,
        )
    assert est.opt_method == 'LM'
    np.testing.assert_array_equal(est.params_convg, lm_par)
    np.testing.assert_array_equal(est.covar_params, lm_cov)

    # 5. Wrong parameter vector shape
    with pytest.raises(ValueError, match="1-D parameter vector of shape"):
        est.optimize_fn(
            method=lambda s, f, g: np.array([0.2, 0.3]),
            verbose=False,
        )
    np.testing.assert_array_equal(est.params_convg, lm_par)

    # 6. Non-finite parameter vector
    with pytest.raises(ValueError, match="non-finite parameter values"):
        est.optimize_fn(
            method=lambda s, f, g: np.array([np.nan]),
            verbose=False,
        )
    np.testing.assert_array_equal(est.params_convg, lm_par)


# ---------------------------------------------------------------------------
# Slice 4B (#534): CalibrationRecord.from_estimation for SciPy & custom fits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('scipy_method', ['scipy', 'least_squares'])
def test_calibration_record_from_estimation_scipy_and_rerun(
        scipy_method: str, tmp_path: Path):
    """Capture a bounded SciPy fit into CalibrationRecord, round-trip, rerun."""
    experiments = make_decay_experiments()
    est = ParameterEstimation(
        first_order_decay,
        [TRUE_CA0, SEED_K],
        experiments,
        optimize_flags=[False, True],
        name_params=['c_A0', 'k'],
    )
    est.optimize_fn(
        method=scipy_method,
        bounds=([LOWER_K], [UPPER_K]),
        optim_options={'max_nfev': 50, 'ftol': SOLVER_TOL, 'xtol': SOLVER_TOL},
        verbose=False,
    )

    sources = [DataSource('synthetic_batch', 'synthetic', 'internal test')]
    model = ModelRevision('first_order_decay', revision='v1')

    record = CalibrationRecord.from_estimation(
        est,
        experiments,
        case=f'decay_{scipy_method}',
        parameter_units={'c_A0': 'mol/L', 'k': '1/s'},
        model=model,
        sources=sources,
    )

    assert record.solver.method == scipy_method
    # Injected 'verbose' and 'bounds' are omitted from solver.options; user
    # options are preserved.
    assert record.solver.options == {
        'max_nfev': 50,
        'ftol': SOLVER_TOL,
        'xtol': SOLVER_TOL,
    }
    assert record.solver.default_options is None
    assert record.solver.outcome['success'] is True
    assert isinstance(record.solver.outcome['status'], int)
    assert isinstance(record.solver.outcome['nfev'], int)
    assert record.solver.outcome['nfev'] >= 1
    assert isinstance(record.solver.outcome['message'], str)
    # Parameter bounds extracted from estimator.optim_options['bounds']
    by_name = {param.name: param for param in record.parameters}
    assert (by_name['c_A0'].lower, by_name['c_A0'].upper) == (None, None)
    assert (by_name['k'].lower, by_name['k'].upper) == (LOWER_K, UPPER_K)

    # Save, reload from JSON, and rerun
    json_path = tmp_path / f'record_{scipy_method}.json'
    record.save(json_path)
    loaded = CalibrationRecord.load(json_path)
    assert loaded == record

    replayed = loaded.rerun(
        first_order_decay, rtol=ESTIMATE_RTOL, atol=ESTIMATE_ATOL)
    np.testing.assert_allclose(
        replayed.params_convg,
        est.params_convg,
        rtol=ESTIMATE_RTOL,
        atol=ESTIMATE_ATOL,
    )


def test_calibration_record_from_estimation_custom_optimizer_and_rerun():
    """Capture a custom-optimizer fit and replay it in-memory and from JSON."""
    experiments = make_decay_experiments()
    est = ParameterEstimation(
        first_order_decay,
        [SEED_K],
        experiments,
        name_params=['k'],
        dx_finitediff=BFGS_FD_STEP,
    )
    est.optimize_fn(
        method=custom_gauss_newton,
        optim_options={'tol': SOLVER_TOL, 'max_iter': MAX_GN_ITER},
        bounds=[(LOWER_K, UPPER_K)],
        verbose=False,
    )

    sources = [DataSource('synthetic_batch', 'synthetic', 'internal test')]
    model = ModelRevision('first_order_decay', revision='v1')
    record = CalibrationRecord.from_estimation(
        est,
        experiments,
        case='decay_custom',
        parameter_units={'k': '1/s'},
        model=model,
        sources=sources,
    )

    assert record.solver.method == 'custom'
    assert record.solver.options == {'tol': SOLVER_TOL, 'max_iter': MAX_GN_ITER}
    assert record.parameters[0].lower == LOWER_K
    assert record.parameters[0].upper == UPPER_K

    # In-memory record retains _custom_optimizer and reruns directly.
    replayed_mem = record.rerun(
        first_order_decay, rtol=ESTIMATE_RTOL, atol=ESTIMATE_ATOL)
    np.testing.assert_allclose(
        replayed_mem.params_convg,
        est.params_convg,
        rtol=ESTIMATE_RTOL,
        atol=ESTIMATE_ATOL,
    )

    # JSON-reloaded record requires optimizer=<callable> in rerun.
    loaded = CalibrationRecord.from_json(record.to_json())
    with pytest.raises(
            ValueError,
            match=r"was fitted with a custom optimizer callable.*optimizer="):
        loaded.rerun(first_order_decay, rtol=ESTIMATE_RTOL, atol=ESTIMATE_ATOL)

    replayed_json = loaded.rerun(
        first_order_decay,
        rtol=ESTIMATE_RTOL,
        atol=ESTIMATE_ATOL,
        optimizer=custom_gauss_newton,
    )
    np.testing.assert_allclose(
        replayed_json.params_convg,
        est.params_convg,
        rtol=ESTIMATE_RTOL,
        atol=ESTIMATE_ATOL,
    )


# ---------------------------------------------------------------------------
# Slice 4C (#536): SciPy least_squares reserved option guard (jac, args, etc.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'reserved_kwargs',
    [
        {'jac': '2-point'},
        {'args': (False,)},
        {'fun': first_order_decay},
        {'x0': [0.2]},
        {'jac': '3-point', 'args': ()},
    ],
)
def test_scipy_least_squares_rejects_reserved_options_with_value_error(
        reserved_kwargs: dict):
    """Passing reserved least_squares keys raises ValueError, not TypeError."""
    experiments = make_decay_experiments()
    est = ParameterEstimation(
        first_order_decay,
        [SEED_K],
        experiments,
        name_params=['k'],
        dx_finitediff=BFGS_FD_STEP,
    )
    # Establish a prior fit to verify rollback when reserved options are rejected.
    prior_par, _, _ = est.optimize_fn(method='LM', verbose=False)

    with pytest.raises(
            ValueError,
            match="optim_options cannot override reserved "
                  "scipy.optimize.least_squares arguments"):
        est.optimize_fn(
            method='scipy',
            optim_options=reserved_kwargs,
            verbose=False,
        )

    with pytest.raises(
            ValueError,
            match="optim_options cannot override reserved "
                  "scipy.optimize.least_squares arguments"):
        est.estimate_parameters(
            method='least_squares',
            verbose=False,
            **reserved_kwargs,
        )

    assert est.opt_method == 'LM'
    np.testing.assert_array_equal(est.params_convg, prior_par)
