"""Exercise the combined #541/#542 fit, diagnostics, and replay contracts.

Real SciPy solves use the existing analytic batch-decay assay. Custom adapters
report only stopping information they actually know; no solver is replaced.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import least_squares

from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision
from test_paramestim_custom_optimizer_and_scipy_records import (
    BFGS_FD_STEP, ESTIMATE_ATOL, ESTIMATE_RTOL, MEAS_STD, SEED_K, SOLVER_TOL,
    first_order_decay, make_decay_experiments,
)


pytestmark = pytest.mark.unit


def scipy_adapter(seed, objective_fun, gradient_fun, *, stopping=None,
                  max_nfev=None, return_pair=False):
    """Solve through the custom interface with real SciPy residual callbacks.

    Parameters
    ----------
    seed : numpy.ndarray
        Free rate constants [1/s], shape ``(num_params,)``.
    objective_fun, gradient_fun : callable
        Public residual [-] and canonical Jacobian [s] callbacks.
    stopping : dict, optional
        Explicit SciPy stopping options with their native units. These are
        applied to the solver and returned as declared tolerance metadata.
    max_nfev : int, optional
        Residual-evaluation limit [-]; one permits only the seed evaluation.
    return_pair : bool, optional
        Return ``(x, metadata)`` instead of a result mapping.

    Returns
    -------
    dict or tuple
        Accepted rate constants [1/s] and actual solver diagnostics.
    """
    def jacobian(params, *args):
        """Return the weighted Jacobian in SciPy's orientation.

        Parameters
        ----------
        params : numpy.ndarray
            Free rate constants [1/s], shape ``(num_params,)``.
        *args
            Residual-array mode forwarded by SciPy with its callback args.

        Returns
        -------
        numpy.ndarray
            Residual Jacobian [s], shape ``(num_residuals, num_params)``.
        """
        return gradient_fun(params, out_array=True).T

    options = {} if stopping is None else dict(stopping)
    result = least_squares(objective_fun, seed, jac=jacobian, args=(True,),
                           max_nfev=max_nfev, **options)
    metadata = dict(result)
    if stopping is not None:
        metadata['tolerances'] = stopping
    if return_pair:
        return result.x, metadata
    return metadata


@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('reported', [False, True])
@pytest.mark.parametrize('return_pair', [False, True])
def test_custom_fit_keeps_only_reported_stopping_metadata(
        named, reported, return_pair):
    """Preserve explicit custom limits without inventing SciPy defaults.

    Parameters
    ----------
    named, reported, return_pair : bool
        Select the custom entry point, explicit stopping metadata, and return
        form. All cases solve the same batch assay.
    """
    estimator = ParameterEstimation(
        first_order_decay, [SEED_K], make_decay_experiments(), name_params=['k'])
    stopping = {'gtol': SOLVER_TOL} if reported else None  # [s], gradient norm
    options = {'stopping': stopping, 'return_pair': return_pair}
    if named:
        options['optimizer'] = scipy_adapter
    _, _, info = estimator.optimize_fn(
        method='custom' if named else scipy_adapter,
        optim_options=options, verbose=False)
    expected = {} if stopping is None else dict(stopping)
    assert info['tolerances'] == expected
    assert info['success']
    assert 'ftol' not in info['tolerances']
    if reported:
        stopping['gtol'] = 0.0  # [s], caller mutation after a completed fit
        assert info['tolerances'] == expected


def test_custom_unsuccessful_fit_warns_and_reaches_identifiability():
    estimator = ParameterEstimation(
        first_order_decay, [SEED_K], make_decay_experiments(), name_params=['k'])
    # A single evaluation cannot update this off-solution seed.
    with pytest.warns(RuntimeWarning, match='Custom optimizer.*without converging'):
        params, _, info = estimator.optimize_fn(
            method=scipy_adapter, optim_options={'max_nfev': 1}, verbose=False)
    np.testing.assert_array_equal(params, [SEED_K])
    assert not info['success']
    report = estimator.get_identifiability()
    assert report.convergence['success'] is False
    assert report.summary().startswith('FIT DID NOT CONVERGE')


def test_custom_convergence_warning_as_error_restores_accepted_fit():
    estimator = ParameterEstimation(
        first_order_decay, [SEED_K], make_decay_experiments(), name_params=['k'])
    params, covariance, _ = estimator.optimize_fn(
        method=scipy_adapter, verbose=False)
    accepted = np.array(estimator.y_model, copy=True)  # [mol/L]
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(RuntimeWarning, match='Custom optimizer'):
            estimator.optimize_fn(
                method=scipy_adapter, optim_options={'max_nfev': 1}, verbose=False)
    np.testing.assert_array_equal(estimator.params_convg, params)
    np.testing.assert_array_equal(estimator.covar_params, covariance)
    np.testing.assert_array_equal(estimator.y_model, accepted)
    assert estimator._custom_optimizer is scipy_adapter


@pytest.mark.parametrize('method', ['scipy', scipy_adapter])
def test_std_fit_diagnostics_and_calibration_record_replay(method):
    """Carry named standard deviations through fitting, diagnostics and replay.

    Parameters
    ----------
    method : str or callable
        Native SciPy or the real custom SciPy adapter.
    """
    canonical = make_decay_experiments()['batch_1'].measurements['c_A']
    measurement = Measurement(x=canonical.x, y=canonical.values, std=MEAS_STD,
                              unit_y='mol/L', basis='molar concentration')
    experiments = {'batch_1': Experiment({'c_A': measurement})}
    estimator = ParameterEstimation(
        first_order_decay, [SEED_K], experiments, name_params=['k'],
        output_names=['c_A'], dx_finitediff=BFGS_FD_STEP)
    params, covariance, info = estimator.estimate_parameters(
        method=method, verbose=False)
    report = estimator.get_identifiability()
    np.testing.assert_allclose(report.covariance, covariance,
                               rtol=ESTIMATE_RTOL, atol=0)
    # Independent d(c_A)/dk = -t*c_A, standardized by the assay sigma.
    concentration = first_order_decay(params, measurement.x)[:, 0]  # [mol/L]
    expected_jacobian = -measurement.x * concentration / MEAS_STD  # [s]
    np.testing.assert_allclose(info['jac'][0], expected_jacobian,
                               rtol=ESTIMATE_RTOL, atol=0)
    before = estimator.get_objective(params, out_array=True).copy()  # [-]
    estimator.check_derivatives()
    np.testing.assert_array_equal(
        estimator.get_objective(params, out_array=True), before)
    record = CalibrationRecord.from_estimation(
        estimator, experiments, case='std_interface_integration',
        parameter_units={'k': '1/s'},
        model=ModelRevision('first_order_decay', revision='analytic-fixture'),
        sources=[DataSource('synthetic_batch', 'synthetic', 'test fixture')])
    loaded = CalibrationRecord.from_json(record.to_json())
    options = {'optimizer': scipy_adapter} if callable(method) else {}
    replay = loaded.rerun(first_order_decay, **options,
                          rtol=ESTIMATE_RTOL, atol=ESTIMATE_ATOL)
    np.testing.assert_allclose(replay.params_convg, params,
                               rtol=ESTIMATE_RTOL, atol=ESTIMATE_ATOL)
    np.testing.assert_array_equal(measurement.std, canonical.uncertainty)


def unused_rate_model(params, time_s):
    """Evaluate decay with one deliberately unidentifiable extra rate.

    Parameters
    ----------
    params : numpy.ndarray
        Rate constants [1/s], shape ``(2,)``; only the first affects the model.
    time_s : numpy.ndarray
        Sampling times [s], shape ``(num_samples,)``.

    Returns
    -------
    numpy.ndarray
        Concentrations [mol/L], shape ``(num_samples, 1)``.
    """
    return first_order_decay(params[:1], time_s)


def test_custom_rank_deficient_fit_keeps_parameters_and_rejects_record():
    """Retain the fit while withholding precision for the unused parameter."""
    experiments = make_decay_experiments()
    estimator = ParameterEstimation(
        unused_rate_model, [SEED_K, SEED_K], experiments,
        name_params=['k', 'unused_rate'])
    with pytest.warns(RuntimeWarning, match='covariance'):
        params, covariance, _ = estimator.optimize_fn(
            method=scipy_adapter, verbose=False)
    assert params.shape == (2,)
    np.testing.assert_array_equal(estimator.params_convg, params)
    assert covariance.shape == (2, 2)
    assert np.isnan(covariance).all()
    with pytest.warns(RuntimeWarning, match='covariance'):
        with pytest.raises(ValueError, match='covariance'):
            CalibrationRecord.from_estimation(
                estimator, experiments, case='unidentifiable_custom_fit',
                parameter_units={'k': '1/s', 'unused_rate': '1/s'},
                model=ModelRevision('unused_rate_model',
                                    revision='analytic-fixture'),
                sources=[DataSource('synthetic_batch', 'synthetic', 'test fixture')])
