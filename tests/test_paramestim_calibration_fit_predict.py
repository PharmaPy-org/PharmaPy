"""Regression tests for issue #78 fit and prediction paths.

Dimensional fixture names carry units, and comments call out dimensionless
intermediates as [-] where normalization or projection removes units. The
missing-cyipopt case blocks cyipopt imports inside an isolated child process,
so it runs whether or not cyipopt is installed; the core CI lane separately
asserts that cyipopt is genuinely absent.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from PharmaPy import ParamEstim
from PharmaPy.Calibration import PCR_calibration


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

_CYIPOPT_IMPORT_BLOCKER = textwrap.dedent('''
    import sys

    class _BlockCyipopt:
        """Reject cyipopt imports in this child process only."""

        @staticmethod
        def find_spec(fullname, path=None, target=None):
            """Reject module specifications for the cyipopt package.

            Parameters
            ----------
            fullname : str
                Fully qualified name of the requested module.
            path : sequence of str or None, optional
                Parent package search path supplied by the import system.
            target : module or None, optional
                Existing module supplied when resolving a reload.

            Returns
            -------
            None
                Signal that unrelated modules should use later finders.

            Raises
            ------
            ModuleNotFoundError
                If cyipopt or one of its submodules is requested.
            """
            if fullname == "cyipopt" or fullname.startswith("cyipopt."):
                raise ModuleNotFoundError(
                    f"No module named {fullname!r}", name=fullname
                )
            return None

    sys.meta_path.insert(0, _BlockCyipopt())

    try:
        import cyipopt
    except ModuleNotFoundError as exc:
        if exc.name != "cyipopt":
            raise AssertionError(f"unexpected blocked module: {exc.name}") from exc
    else:
        raise AssertionError("cyipopt import blocker did not reject the backend")
    ''')


def _run_with_cyipopt_blocked(script, tmp_path):
    """Run Python source with cyipopt blocked in the child process.

    Parameters
    ----------
    script : str
        Python source to execute in the child process.
    tmp_path : pathlib.Path
        Temporary directory used for the Matplotlib configuration cache.

    Returns
    -------
    subprocess.CompletedProcess
        Completed child-process result with captured text output.

    Notes
    -----
    ``PharmaPy.ParamEstim`` sets ``have_cyipopt`` once, at import time. The
    child installs the blocker before that import, so the module takes its
    missing-cyipopt branch without altering the parent interpreter.
    """
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path)
    blocked_script = f"{_CYIPOPT_IMPORT_BLOCKER}\n{script}"
    return subprocess.run(
        [sys.executable, "-c", blocked_script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_parameter_estimation_reports_missing_cyipopt(tmp_path):
    """IPOPT selection without cyipopt reports how to enable IPOPT fitting."""
    script = textwrap.dedent('''
        import numpy as np

        from PharmaPy import ParamEstim

        if ParamEstim.have_cyipopt is not False:
            raise AssertionError("blocked cyipopt was reported as available")

        time_s = np.array([0.0, 1.0])  # [s]
        observed_concentration_mol_l = np.array([0.0, 1.0])  # [mol/L]

        def linear_model(params, x_data_s):
            """Return concentration [mol/L] from rate [mol/L/s], time [s]."""
            return params[0] * x_data_s

        estimator = ParamEstim.ParameterEstimation(
            linear_model,
            param_seed=np.array([1.0]),  # [mol/L/s]
            x_data=time_s,
            y_data=observed_concentration_mol_l,
            name_params=["rate_mol_l_s"],
        )

        try:
            estimator.optimize_fn(method="IPOPT", verbose=False)
        except ImportError as exc:
            if "cyipopt is an optional import" not in str(exc):
                raise AssertionError(str(exc)) from exc
        else:
            raise AssertionError("IPOPT selection unexpectedly succeeded")
        ''')
    result = _run_with_cyipopt_blocked(script, tmp_path)

    assert result.returncode == 0, result.stderr


def test_parameter_estimation_assembles_ipopt_result_info():
    """Exercise the solver-independent IPOPT post-solve result contract.

    The test calls the real objective, gradient, and covariance methods in the
    same order as the IPOPT result path. Time is [s], concentration is [mol/L],
    and the fitted rate is [mol/L/s].
    """
    time_s = np.array([0.0, 1.0, 2.0])
    rate_seed_mol_l_s = 1.0
    rate_mol_l_s = 2.0
    y_model_mol_l = rate_mol_l_s * time_s
    residual_offset_mol_l = np.array([0.10, -0.05, 0.20])
    y_obs_mol_l = y_model_mol_l + residual_offset_mol_l

    def linear_model(params, x_data_s):
        """Return concentration [mol/L] from rate [mol/L/s] and time [s]."""
        return params[0] * x_data_s

    def linear_jacobian(params, x_data_s):
        """Return d(concentration)/d(rate) sensitivities with units [s]."""
        return x_data_s[np.newaxis, :]

    estimator = ParamEstim.ParameterEstimation(
        linear_model,
        param_seed=[rate_seed_mol_l_s],
        x_data=time_s,
        y_data=y_obs_mol_l,
        name_params=["rate_mol_l_s"],
        jac_fun=linear_jacobian,
    )

    opt_par_mol_l_s = np.array([rate_mol_l_s])  # [mol/L/s]
    # Match the real IPOPT callback followed by the production,
    # solver-independent result assembly. Issue #78 used the invalid ``base``
    # keyword inside this helper's objective call.
    estimator.get_objective(opt_par_mol_l_s)
    info = estimator.assemble_solver_info(opt_par_mol_l_s)
    estimator.info_opt = info
    covar_rate = estimator.get_covariance()
    y_model_mol_l_actual = estimator.resid_runs[0] + estimator.y_data[0]

    # The default identity weight matrix leaves the [mol/L] residual and [s]
    # sensitivity values numerically unchanged after sigma_inv weighting.
    expected_weighted_residuals = -residual_offset_mol_l
    expected_weighted_jacobian_s = time_s[np.newaxis, :]

    np.testing.assert_allclose(opt_par_mol_l_s, [rate_mol_l_s])
    np.testing.assert_allclose(info["fun"], expected_weighted_residuals)
    np.testing.assert_allclose(info["jac"], expected_weighted_jacobian_s)
    np.testing.assert_allclose(y_model_mol_l_actual.ravel(), y_model_mol_l)
    # Covariance entries correspond to rate variance units [(mol/L/s)^2].
    assert covar_rate.shape == (1, 1)


def test_pcr_predict_uses_training_centering_for_single_new_spectrum():
    """Predict a single absorbance spectrum [AU] with training statistics."""
    # Calibration predictor rows are spectra [AU] at three wavelengths.
    spectra_au = np.array([
        [0.20, 1.10, 2.40],
        [0.45, 1.35, 2.95],
        [0.80, 1.85, 3.45],
        [1.10, 2.10, 4.05],
    ])
    # Response concentrations are [g/L].
    concentration_g_l = np.array([1.2, 1.8, 2.6, 3.1])

    num_comp = 2  # number of retained principal components [-]
    calibration = PCR_calibration(spectra_au, num_comp=num_comp,
                                  standardize=True)
    calibration.get_regression(concentration_g_l, num_comp=num_comp)

    # Single prediction spectrum is in the same absorbance units [AU].
    new_spectrum_au = np.array([[0.70, 1.70, 3.20]])
    prediction_g_l = calibration.predict(new_spectrum_au)

    # Predictor centering uses the training absorbance statistics [AU], and
    # division by the training standard deviation makes the predictors [-].
    training_mean_au = spectra_au.mean(axis=0)
    training_std_au = spectra_au.std(axis=0)
    centered_new = (
        (new_spectrum_au - training_mean_au) / training_std_au
    )
    # SVD loadings and principal-component scores are dimensionless [-].
    loadings = calibration.svd_dict["V"][:, :num_comp]
    scores = centered_new @ loadings
    # Regression coefficients convert dimensionless scores [-] to [g/L], and
    # y_means is the response offset [g/L].
    regression_coeff_g_l = calibration.regression_coeff
    response_offset_g_l = calibration.y_means
    expected_g_l = scores @ regression_coeff_g_l + response_offset_g_l

    assert np.all(np.isfinite(prediction_g_l))
    np.testing.assert_allclose(prediction_g_l, expected_g_l)
