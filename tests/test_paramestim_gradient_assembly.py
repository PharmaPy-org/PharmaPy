"""Regression tests for fixed-parameter and IPOPT gradient assembly.

The fixtures use concentration responses [mol/L] over time [s]. Parameter
units are stated beside each setup value because the gradients combine
state sensitivities, residual weights, and residual ordering.
"""

import numpy as np
import pytest

from PharmaPy import ParamEstim


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "param_seed, optimize_flags, expected_params, expected_jacobian",
    [
        (
            np.array([1.5, 0.5]),  # [mol/L/s], [mol/L]
            [True, False],
            np.array([2.0]),  # [mol/L/s]
            np.array([[0.0, 1.0, 2.0]]),  # [s]
        ),
        (
            np.array([2.0, 1.0]),  # [mol/L/s], [mol/L]
            [False, True],
            np.array([0.5]),  # [mol/L]
            np.array([[1.0, 1.0, 1.0]]),  # [-]
        ),
    ],
)
def test_lm_optimization_reconstructs_fixed_parameters(
        param_seed, optimize_flags, expected_params, expected_jacobian):
    """Fixed parameters keep full model positions through LM callbacks."""
    time_s = np.array([0.0, 1.0, 2.0])  # [s]
    observed_conc_mol_l = 2.0 * time_s + 0.5  # [mol/L]

    def affine_model(params, x_data_s):
        """Return concentration [mol/L] from rate [mol/L/s] and offset [mol/L]."""
        rate_mol_l_s, offset_mol_l = params  # [mol/L/s], [mol/L]
        return rate_mol_l_s * x_data_s + offset_mol_l

    estimator = ParamEstim.ParameterEstimation(
        affine_model,
        param_seed=param_seed,
        x_data=time_s,
        y_data=observed_conc_mol_l,
        optimize_flags=optimize_flags,
        name_params=["rate_mol_l_s", "offset_mol_l"],
    )
    lm_options = {"max_fun_eval": 5}  # [-]
    optimized_params, covar_params, info = estimator.optimize_fn(
        method="LM", verbose=False, optim_options=lm_options)

    # The affine fixture makes finite-difference sensitivities exact up to
    # roundoff, while the LM parameter solve also carries convergence slack.
    parameter_atol = 1e-7  # [mol/L/s] or [mol/L]
    jacobian_atol = 1e-10  # [s] or [-]
    response_atol = 1e-7  # [mol/L]
    np.testing.assert_allclose(optimized_params, expected_params, rtol=0.0,
                               atol=parameter_atol)
    np.testing.assert_allclose(info["jac"], expected_jacobian, rtol=0.0,
                               atol=jacobian_atol)
    # optimize_fn rebuilds the fitted response as residual plus data. This
    # affine fixture is exactly representable, so the stored profile matches
    # the observations to LM convergence slack.
    np.testing.assert_allclose(
        estimator.y_model[0].ravel(), observed_conc_mol_l,
        rtol=0.0, atol=response_atol,
    )
    assert covar_params.shape == (1, 1)


def test_scalar_gradient_recomputes_residuals_for_requested_params():
    """The scalar optimizer gradient is evaluated at its input parameters."""
    time_s = np.array([1.0, 2.0, 4.0])  # [s]
    rate_mol_l_s = 2.0  # [mol/L/s]
    acceleration_mol_l_s2 = 3.0  # [mol/L/s**2]
    params = np.array([rate_mol_l_s, acceleration_mol_l_s2])  # [mol/L/s], [mol/L/s**2]
    trial_params = params + np.array([0.5, 0.5])  # [mol/L/s], [mol/L/s**2]

    model_conc_mol_l = np.column_stack(  # [mol/L]
        (
            rate_mol_l_s * time_s,
            acceleration_mol_l_s2 * time_s**2,
        )
    )
    residuals_mol_l = np.array(  # [mol/L]
        [
            [6.0, -3.0],
            [8.0, -6.0],
            [10.0, -9.0],
        ]
    )
    observed_conc_mol_l = model_conc_mol_l - residuals_mol_l  # [mol/L]
    weight_matrix = np.diag([4.0, 9.0])  # [(mol/L)**2]

    def two_state_model(params_eval, x_data_s):
        """Return two concentration states [mol/L] at each time [s]."""
        rate_eval_mol_l_s, accel_eval_mol_l_s2 = params_eval  # [mol/L/s], [mol/L/s**2]
        return np.column_stack(
            (
                rate_eval_mol_l_s * x_data_s,
                accel_eval_mol_l_s2 * x_data_s**2,
            )
        )

    def two_state_jacobian(params_eval, x_data_s):
        """Return state-major sensitivities [s] and [s**2]."""
        sens_state_0 = np.column_stack(
            (
                x_data_s,
                np.zeros_like(x_data_s),
            )
        )  # [s], [s**2]
        sens_state_1 = np.column_stack(
            (
                np.zeros_like(x_data_s),
                x_data_s**2,
            )
        )  # [s], [s**2]
        return np.vstack((sens_state_0, sens_state_1))

    estimator = ParamEstim.ParameterEstimation(
        two_state_model,
        param_seed=params,
        x_data=time_s,
        y_data=observed_conc_mol_l,
        jac_fun=two_state_jacobian,
        weight_matrix=weight_matrix,
        name_params=["rate_mol_l_s", "acceleration_mol_l_s2"],
        name_states=["linear_conc_mol_l", "quadratic_conc_mol_l"],
    )

    # Reproduce the callback order that exposed issue #83 using the public
    # objective and gradient contracts directly: a trial objective is followed
    # by a gradient request at a different parameter vector.
    estimator.get_objective(trial_params)
    gradient = estimator.get_gradient(params)

    # These hand-computed gradients use residuals at params, not at the trial
    # objective point.
    expected_gradient = np.array([15.5, -19.0])  # [s*L/mol], [s**2*L/mol]
    np.testing.assert_allclose(gradient, expected_gradient,
                               rtol=0.0, atol=1e-12)


def test_scalar_gradient_recomputes_callback_sensitivities():
    """Scalar optimizer gradients refresh parameter-dependent sensitivities."""
    time_s = np.array([1.0, 2.0, 4.0])  # [s]
    rate_mol_l_s = 2.0  # [mol/L/s]
    acceleration_mol_l_s2 = 3.0  # [mol/L/s**2]
    params = np.array([rate_mol_l_s, acceleration_mol_l_s2])  # [mol/L/s], [mol/L/s**2]
    trial_params = params + np.array([0.5, 0.5])  # [mol/L/s], [mol/L/s**2]

    model_conc_mol_l = np.column_stack(  # [mol/L]
        (
            rate_mol_l_s**2 * time_s,
            acceleration_mol_l_s2**2 * time_s**2,
        )
    )
    residuals_mol_l = np.array(  # [mol/L]
        [
            [6.0, -3.0],
            [8.0, -6.0],
            [10.0, -9.0],
        ]
    )
    observed_conc_mol_l = model_conc_mol_l - residuals_mol_l  # [mol/L]
    weight_matrix = np.diag([4.0, 9.0])  # [(mol/L)**2]

    def two_state_model_with_sens(params_eval, x_data_s):
        """Return concentration states [mol/L] and sensitivities."""
        rate_eval_mol_l_s, accel_eval_mol_l_s2 = params_eval  # [mol/L/s], [mol/L/s**2]
        states_mol_l = np.column_stack(  # [mol/L]
            (
                rate_eval_mol_l_s**2 * x_data_s,
                accel_eval_mol_l_s2**2 * x_data_s**2,
            )
        )
        sens_state_0 = np.column_stack(
            (
                2.0 * rate_eval_mol_l_s * x_data_s,
                np.zeros_like(x_data_s),
            )
        )  # [s], [s**2]
        sens_state_1 = np.column_stack(
            (
                np.zeros_like(x_data_s),
                2.0 * accel_eval_mol_l_s2 * x_data_s**2,
            )
        )  # [s], [s**2]
        sensitivities = np.vstack((sens_state_0, sens_state_1))  # [s], [s**2]
        return states_mol_l, sensitivities

    estimator = ParamEstim.ParameterEstimation(
        two_state_model_with_sens,
        param_seed=params,
        x_data=time_s,
        y_data=observed_conc_mol_l,
        weight_matrix=weight_matrix,
        name_params=["rate_mol_l_s", "acceleration_mol_l_s2"],
        name_states=["linear_conc_mol_l", "quadratic_conc_mol_l"],
    )

    estimator.get_objective(trial_params)
    gradient = estimator.get_gradient(params)

    expected_gradient = np.array([62.0, -114.0])  # [s*L/mol], [s**2*L/mol]
    np.testing.assert_allclose(gradient, expected_gradient,
                               rtol=0.0, atol=1e-12)


def test_scalar_gradient_uses_weighted_state_major_residuals():
    """The scalar gradient matches weighted residual objective units."""
    time_s = np.array([1.0, 2.0, 4.0])  # [s]
    rate_mol_l_s = 2.0  # [mol/L/s]
    acceleration_mol_l_s2 = 3.0  # [mol/L/s**2]
    params = np.array([rate_mol_l_s, acceleration_mol_l_s2])  # [mol/L/s], [mol/L/s**2]

    model_conc_mol_l = np.column_stack(  # [mol/L]
        (
            rate_mol_l_s * time_s,
            acceleration_mol_l_s2 * time_s**2,
        )
    )
    residuals_mol_l = np.array(  # [mol/L]
        [
            [6.0, -3.0],
            [8.0, -6.0],
            [10.0, -9.0],
        ]
    )
    observed_conc_mol_l = model_conc_mol_l - residuals_mol_l  # [mol/L]
    weight_matrix = np.diag([4.0, 9.0])  # [(mol/L)**2]

    def two_state_model(params_eval, x_data_s):
        """Return two concentration states [mol/L] at each time [s]."""
        rate_eval_mol_l_s, accel_eval_mol_l_s2 = params_eval  # [mol/L/s], [mol/L/s**2]
        return np.column_stack(
            (
                rate_eval_mol_l_s * x_data_s,
                accel_eval_mol_l_s2 * x_data_s**2,
            )
        )

    def two_state_jacobian(params_eval, x_data_s):
        """Return state-major sensitivities [s] and [s**2]."""
        sens_state_0 = np.column_stack(
            (
                x_data_s,
                np.zeros_like(x_data_s),
            )
        )  # [s], [s**2]
        sens_state_1 = np.column_stack(
            (
                np.zeros_like(x_data_s),
                x_data_s**2,
            )
        )  # [s], [s**2]
        return np.vstack((sens_state_0, sens_state_1))

    estimator = ParamEstim.ParameterEstimation(
        two_state_model,
        param_seed=params,
        x_data=time_s,
        y_data=observed_conc_mol_l,
        jac_fun=two_state_jacobian,
        weight_matrix=weight_matrix,
        name_params=["rate_mol_l_s", "acceleration_mol_l_s2"],
        name_states=["linear_conc_mol_l", "quadratic_conc_mol_l"],
    )

    estimator.get_objective(params)
    gradient = estimator.get_gradient(params)

    expected_gradient = np.array([15.5, -19.0])  # [s*L/mol], [s**2*L/mol]
    np.testing.assert_allclose(gradient, expected_gradient)
