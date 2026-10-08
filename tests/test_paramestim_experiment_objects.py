"""Experiment and Measurement objects in parameter estimation (issue #346).

Scope: the public ``Experiment``/``Measurement`` data contract of
``ParameterEstimation``; its single compile path shared with legacy arrays,
lists and dictionaries; residual packing (``get_residual_layout``);
replicate mapping on multiset-union grids; per-measurement uncertainty
scaling; validation messages; and caller-data isolation.

Fixtures: an analytic first-order decay A -> B with an adiabatic
temperature rise, whose states and rate sensitivities have closed forms,
and a two-output model linear in its rate for closed-form generalized least
squares (GLS). Expected values come from those closed forms and from
hand-written layouts, never from estimator internals. Everything runs in
milliseconds with PharmaPy's Levenberg-Marquardt (LM) solver; no optional
backend is used.
"""

import re

import numpy as np
import pandas as pd
import pytest

from PharmaPy.ParamEstim import (Experiment, Measurement,
                                 MultipleCurveResolution, ParameterEstimation)

pytestmark = pytest.mark.unit

TRUE_RATE = 0.5  # [1/s], rate that generates the synthetic data
SEED_RATE = 0.3  # [1/s], non-optimal evaluation point and LM seed
# Closed-form and production values are the same IEEE expressions evaluated
# on differently shaped arrays; a few ulps of O(300) values bound the
# difference, far below the O(1e-2) offsets that distinguish layouts.
ROUNDOFF_RTOL = 1e-12  # [-]
ROUNDOFF_ATOL = 1e-12  # [residual unit]
# The LM default termination tolerances stop within 1e-8 relative of the
# optimum for these one-parameter fits; 1e-6 leaves a two-decade margin.
FIT_RTOL = 1e-6  # [-]
# Default finite differences are forward steps h = 1e-3 * rate, whose
# truncation error is h/2 * |d2y/dk2|; the bound below allows twice that.
FD_TRUNCATION_FACTOR = 1.0  # [-], multiplies h * |d2y/dk2|
# Forward-difference roundoff eps * |y| / h is below 1e-9 for |y| <= 350 K
# and h = 3e-4 1/s; 1e-8 leaves a decade of margin.
FD_ROUNDOFF_ATOL = 1e-8  # [state unit / (1/s)]
FD_STEP_FACTOR = 1e-3  # [-], sqrt(1e-6) relative step of dx_jac_p


def decay_model(params, time_s, initial_mol_l, product_mol_l=0.0,
                adiabatic_rise_k=0.0, initial_temp_k=300.0):
    """Evaluate first-order A -> B with an adiabatic temperature rise.

    Parameters
    ----------
    params : array_like
        Rate constant k [1/s], shape (1,).
    time_s : numpy.ndarray
        Model times [s], shape (n_times,).
    initial_mol_l : float
        Initial concentration of A [mol/L].
    product_mol_l : float, optional
        Initial concentration of B [mol/L].
    adiabatic_rise_k : float, optional
        Temperature rise at full conversion [K].
    initial_temp_k : float, optional
        Initial temperature [K].

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 3): c_A [mol/L], c_B [mol/L], T [K].
    """
    conversion = 1 - np.exp(-params[0] * time_s)  # [-]
    return np.column_stack((
        initial_mol_l * (1 - conversion),
        product_mol_l + initial_mol_l * conversion,
        initial_temp_k + adiabatic_rise_k * conversion))


def decay_sensitivity(params, time_s, initial_mol_l, product_mol_l=0.0,
                      adiabatic_rise_k=0.0, initial_temp_k=300.0):
    """Return analytic rate sensitivities of ``decay_model``, state-major.

    Parameters
    ----------
    params, time_s, initial_mol_l, product_mol_l
        As for ``decay_model``.
    adiabatic_rise_k, initial_temp_k
        As for ``decay_model``.

    Returns
    -------
    numpy.ndarray
        Shape (3 * n_times, 1): dc_A/dk, dc_B/dk [mol/L*s], dT/dk [K*s].
    """
    decay = time_s * np.exp(-params[0] * time_s)  # [s]
    return np.concatenate((-initial_mol_l * decay, initial_mol_l * decay,
                           adiabatic_rise_k * decay))[:, np.newaxis]


def decay_point(rate, time, field, args, kwargs, derivative=0):
    """Evaluate one state or rate derivative of the decay model in closed form.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].
    time : float
        Time [s].
    field : int
        0 for c_A, 1 for c_B, 2 for T.
    args : tuple
        ``(initial_mol_l,)`` [mol/L].
    kwargs : dict
        Remaining ``decay_model`` keywords.
    derivative : int, optional
        Order of the derivative with respect to k (0, 1 or 2).

    Returns
    -------
    float
        State value [mol/L or K] or its k-derivative [state unit * s**order].
    """
    initial, = args
    product = kwargs.get('product_mol_l', 0.0)  # [mol/L]
    rise = kwargs.get('adiabatic_rise_k', 0.0)  # [K]
    temp0 = kwargs.get('initial_temp_k', 300.0)  # [K]
    decay = np.exp(-rate * time)  # [-]
    if derivative == 0:
        return [initial * decay, product + initial * (1 - decay),
                temp0 + rise * (1 - decay)][field]
    # d^n(exp(-k t))/dk^n = (-t)^n exp(-k t); conversion is 1 - exp(-k t).
    term = (-time) ** derivative * decay  # [s**n]
    return [initial * term, -initial * term, -rise * term][field]


# Two experiments with distinct callback arguments, unequal grids, a
# single-point temperature, an omitted c_B column in 'cold', and a missing
# (NaN) c_B observation in 'hot'. Offsets are added to the noise-free data.
FIXTURE = {
    'hot': {
        'args': (2.0,),  # [mol/L]
        # Initial B [mol/L] and adiabatic rise [K].
        'kwargs': {'product_mol_l': 0.5, 'adiabatic_rise_k': 40.0},
        'measurements': {
            # name: (field, times [s], offsets [mol/L or K], units)
            'c_A': (0, [0.0, 1.0, 2.0, 4.0], [0.02, -0.01, 0.015, -0.005],
                    'mol/L'),
            'c_B': (1, [1.0, 3.0, 4.0], [0.01, np.nan, -0.02], 'mol/L'),
            'T': (2, [2.0], [0.3], 'K'),
        },
    },
    'cold': {
        'args': (1.0,),  # [mol/L]
        'kwargs': {'adiabatic_rise_k': 10.0, 'initial_temp_k': 290.0},  # [K]
        'measurements': {
            'c_A': (0, [0.5, 1.5, 3.0], [-0.01, 0.005, 0.01], 'mol/L'),
            'T': (2, [1.5], [-0.2], 'K'),
        },
    },
}

# Hand-written packing: experiment, then measurement column (first
# appearance c_A, c_B, T), then model-grid sample. 'hot' grid is the union
# [0, 1, 2, 3, 4] s, 'cold' grid [0.5, 1.5, 3] s.
EXPECTED_LAYOUT = (
    [('hot', 'c_A', 0, t, o) for t, o in zip([0, 1, 2, 3, 4],
                                             [1, 1, 1, 0, 1])]
    + [('hot', 'c_B', 1, t, o) for t, o in zip([0, 1, 2, 3, 4],
                                               [0, 1, 0, 0, 1])]
    + [('hot', 'T', 2, t, o) for t, o in zip([0, 1, 2, 3, 4],
                                             [0, 0, 1, 0, 0])]
    + [('cold', 'c_A', 0, t, o) for t, o in zip([0.5, 1.5, 3], [1, 1, 1])]
    + [('cold', 'c_B', 1, t, o) for t, o in zip([0.5, 1.5, 3], [0, 0, 0])]
    + [('cold', 'T', 2, t, o) for t, o in zip([0.5, 1.5, 3], [0, 1, 0])])
NUM_OBSERVED = 11  # [-], 4 + 2 + 1 (hot) + 3 + 1 (cold)


def observation(experiment, name, time, offsets=True):
    """Return one synthetic observation in closed form.

    Parameters
    ----------
    experiment, name : str
        Keys of ``FIXTURE``.
    time : float
        Sample time [s].
    offsets : bool, optional
        Add the fixture offset; False gives noise-free data.

    Returns
    -------
    float
        Observation [mol/L or K]; NaN if missing.
    """
    spec = FIXTURE[experiment]
    field, times, shifts, _ = spec['measurements'][name]
    shift = shifts[times.index(time)]  # [mol/L or K]
    if np.isnan(shift):
        return np.nan
    return (decay_point(TRUE_RATE, time, field, spec['args'], spec['kwargs'])
            + (shift if offsets else 0.0))


def build_experiments(offsets=True, reverse=False, uncertainty=None):
    """Build the fixture as Experiment objects.

    Parameters
    ----------
    offsets : bool, optional
        Add the fixture offsets to the noise-free observations.
    reverse : bool, optional
        Reverse experiment and measurement insertion orders.
    uncertainty : dict, optional
        Measurement name -> standard deviation [mol/L or K].

    Returns
    -------
    dict
        Experiment name -> Experiment.
    """
    order = (lambda items: list(reversed(list(items)))) if reverse else list
    experiments = {}
    for name, spec in order(FIXTURE.items()):
        measurements = {}
        for meas, (field, times, _, units) in order(
                spec['measurements'].items()):
            # [mol/L or K]
            values = [observation(name, meas, t, offsets) for t in times]
            measurements[meas] = Measurement(
                field, times, values, units=units,
                uncertainty=None if uncertainty is None else uncertainty[meas])
        experiments[name] = Experiment(measurements, args=spec['args'],
                                       kwargs=spec['kwargs'])
    return experiments


def expected_residuals(rate, scale=None):
    """Return closed-form residuals in the hand-written packing order.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].
    scale : dict, optional
        Measurement name -> standard deviation dividing the residual.

    Returns
    -------
    numpy.ndarray
        Model minus data [mol/L or K], or [-] when scaled; zero where
        unobserved.
    """
    out = []
    for experiment, name, field, time, observed in EXPECTED_LAYOUT:
        if not observed:
            out.append(0.0)
            continue
        spec = FIXTURE[experiment]
        residual = (decay_point(rate, time, field, spec['args'],
                                spec['kwargs'])
                    - observation(experiment, name, time))  # [mol/L or K]
        out.append(residual / (1.0 if scale is None else scale[name]))
    return np.array(out)


def expected_jacobian(rate, order=1, scale=None):
    """Return closed-form rate derivatives of the residuals.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].
    order : int, optional
        Derivative order (1 for the Jacobian, 2 for curvature).
    scale : dict, optional
        Measurement name -> standard deviation dividing the derivative.

    Returns
    -------
    numpy.ndarray
        Derivatives [state unit * s**order], zero where unobserved.
    """
    out = []
    for experiment, name, field, time, observed in EXPECTED_LAYOUT:
        spec = FIXTURE[experiment]
        value = decay_point(rate, time, field, spec['args'], spec['kwargs'],
                            derivative=order) if observed else 0.0
        out.append(value / (1.0 if scale is None else scale[name]))
    return np.array(out)


def keyed(estimator, values):
    """Key residual-like values by experiment, measurement and x.

    Parameters
    ----------
    estimator : ParameterEstimation
        Estimator whose ``get_residual_layout`` describes ``values``.
    values : numpy.ndarray
        Residuals or Jacobian rows in packing order.

    Returns
    -------
    dict
        ``(experiment, measurement, x) -> value``.
    """
    layout = estimator.get_residual_layout()
    keys = zip(layout['experiment'], layout['measurement'], layout['x'])
    return dict(zip(keys, values))


def test_residual_layout_matches_hand_written_packing():
    estimator = ParameterEstimation(decay_model, [SEED_RATE],
                                    build_experiments())
    layout = estimator.get_residual_layout()
    expected = pd.DataFrame(EXPECTED_LAYOUT, columns=[
        'experiment', 'measurement', 'field', 'x', 'observed'])
    expected = expected.astype({'x': float, 'observed': bool})
    pd.testing.assert_frame_equal(layout, expected)
    residuals = estimator.get_objective([SEED_RATE], out_array=True)
    assert len(layout) == len(residuals)
    masks = np.concatenate([mask.T.ravel() for mask in estimator.x_masks])
    np.testing.assert_array_equal(layout['observed'].to_numpy(), masks)
    assert estimator.num_data_total == NUM_OBSERVED
    assert estimator.num_data == [7, 4]
    assert estimator.experim_names == ['hot', 'cold']
    assert estimator.measured_ind == [0, 1, 2]
    assert estimator.name_states == ['c_A', 'c_B', 'T']
    np.testing.assert_array_equal(estimator.x_data[0], [0., 1., 2., 3., 4.])
    np.testing.assert_array_equal(estimator.x_model[1], [0.5, 1.5, 3.])


@pytest.mark.parametrize('use_jac', [True, False], ids=['jac_fun', 'fd'])
def test_objective_and_jacobian_match_closed_form(use_jac):
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE], build_experiments(),
        jac_fun=decay_sensitivity if use_jac else None)
    residual = expected_residuals(SEED_RATE)  # [mol/L or K]
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True), residual,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)
    assert estimator.get_objective([SEED_RATE]) == pytest.approx(
        0.5 * residual @ residual, rel=ROUNDOFF_RTOL)
    jacobian = estimator.get_gradient([SEED_RATE], out_array=True)
    assert jacobian.shape == (1, len(EXPECTED_LAYOUT))
    expected = expected_jacobian(SEED_RATE)  # [state unit * s]
    if use_jac:
        np.testing.assert_allclose(jacobian[0], expected,
                                   rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)
    else:
        step = FD_STEP_FACTOR * SEED_RATE  # [1/s]
        bound = (FD_TRUNCATION_FACTOR * step
                 * np.abs(expected_jacobian(SEED_RATE, order=2))
                 + FD_ROUNDOFF_ATOL)  # [state unit * s]
        assert np.all(np.abs(jacobian[0] - expected) <= bound)
        assert np.all(jacobian[0][expected == 0.0] == 0.0)


def test_reordered_mappings_preserve_objective_jacobian_and_fit():
    reference = ParameterEstimation(decay_model, [SEED_RATE],
                                    build_experiments(),
                                    jac_fun=decay_sensitivity)
    reordered = ParameterEstimation(decay_model, [SEED_RATE],
                                    build_experiments(reverse=True),
                                    jac_fun=decay_sensitivity)
    assert reordered.experim_names == ['cold', 'hot']
    assert reordered.get_residual_layout()['measurement'].unique().tolist() \
        == ['T', 'c_A', 'c_B']
    for method, out_array in (('get_objective', True),
                              ('get_gradient', True)):
        values = [np.ravel(getattr(estimator, method)([SEED_RATE],
                                                      out_array=out_array))
                  for estimator in (reference, reordered)]
        assert keyed(reference, values[0]) == pytest.approx(
            keyed(reordered, values[1]), rel=ROUNDOFF_RTOL,
            abs=ROUNDOFF_ATOL)
    assert reordered.get_objective([SEED_RATE]) == pytest.approx(
        reference.get_objective([SEED_RATE]), rel=ROUNDOFF_RTOL)
    fitted = [estimator.optimize_fn(verbose=False)[0]
              for estimator in (reference, reordered)]
    np.testing.assert_allclose(fitted[1], fitted[0], rtol=FIT_RTOL)


@pytest.mark.parametrize('use_jac', [True, False], ids=['jac_fun', 'fd'])
def test_noise_free_fit_recovers_true_rate(use_jac):
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE], build_experiments(offsets=False),
        jac_fun=decay_sensitivity if use_jac else None)
    fitted, _, info = estimator.optimize_fn(verbose=False)
    np.testing.assert_allclose(fitted, [TRUE_RATE], rtol=FIT_RTOL)
    assert info['jac'].shape == (1, len(EXPECTED_LAYOUT))
    unobserved = ~np.array([row[4] for row in EXPECTED_LAYOUT], dtype=bool)
    assert np.all(info['fun'][unobserved] == 0.0)
    assert np.all(info['jac'][:, unobserved] == 0.0)


def test_uncertainty_scales_mixed_units_to_dimensionless_residuals():
    sigma = {'c_A': 0.01, 'c_B': 0.02, 'T': 0.5}  # [mol/L], [mol/L], [K]
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE], build_experiments(uncertainty=sigma),
        jac_fun=decay_sensitivity)
    scaled = expected_residuals(SEED_RATE, scale=sigma)  # [-]
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True), scaled,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)
    assert estimator.get_objective([SEED_RATE]) == pytest.approx(
        0.5 * scaled @ scaled, rel=ROUNDOFF_RTOL)
    np.testing.assert_allclose(
        estimator.get_gradient([SEED_RATE], out_array=True)[0],
        expected_jacobian(SEED_RATE, scale=sigma),
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)
    assert estimator.num_data_total == NUM_OBSERVED
    # Non-vacuity: the unscaled temperature residual differs from the
    # scaled one by the factor 1/sigma_T = 2.
    unscaled = expected_residuals(SEED_RATE)  # [mol/L or K]
    assert not np.allclose(unscaled, scaled)


def test_per_sample_uncertainty_ignores_missing_samples():
    times = [0.0, 1.0, 2.0]  # [s]
    values = [2.0, np.nan, 0.8]  # [mol/L], middle sample missing
    std = [0.1, -1.0, 0.4]  # [mol/L]; the negative entry is unobserved
    experiment = Experiment([Measurement(0, times, values, uncertainty=std)],
                            args=(2.0,))
    estimator = ParameterEstimation(decay_model, [SEED_RATE], experiment)
    # [mol/L]
    model = [decay_point(SEED_RATE, t, 0, (2.0,), {}) for t in times]
    expected = [(model[0] - 2.0) / 0.1, 0.0, (model[2] - 0.8) / 0.4]  # [-]
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True), expected,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)
    assert estimator.num_data_total == 2


def test_replicates_map_to_multiset_union_grid():
    # 'a' has three replicates at x = 1 s and 'b' two, so the grid holds
    # 1 s three times; 'b' leaves the third copy unobserved.
    first = Measurement(0, [0., 1., 1., 1., 2.], [10., 11., 12., 13., 14.])
    second = Measurement(1, [1., 1., 3.], [21., 22., 23.])
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE],
        {'run': Experiment({'a': first, 'b': second}, args=(2.0,))})
    np.testing.assert_array_equal(estimator.x_model[0],
                                  [0., 1., 1., 1., 2., 3.])
    np.testing.assert_array_equal(
        estimator.x_masks[0],
        [[True, False], [True, True], [True, True], [True, False],
         [True, False], [False, True]])
    np.testing.assert_array_equal(
        estimator.y_data[0],
        [[10., np.nan], [11., 21.], [12., 22.], [13., np.nan], [14., np.nan],
         [np.nan, 23.]])
    grid = [0., 1., 1., 1., 2., 3.]  # [s]
    c_a = [decay_point(SEED_RATE, t, 0, (2.0,), {}) for t in grid]  # [mol/L]
    c_b = [decay_point(SEED_RATE, t, 1, (2.0,), {}) for t in grid]  # [mol/L]
    expected = np.array(
        [c_a[0] - 10, c_a[1] - 11, c_a[2] - 12, c_a[3] - 13, c_a[4] - 14, 0.,
         0., c_b[1] - 21, c_b[2] - 22, 0., 0., c_b[5] - 23])  # [mol/L]
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True), expected,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)
    assert estimator.num_data_total == 8


def test_replicate_measurements_of_one_field_share_the_model_column():
    times = [1.0, 2.0]  # [s]
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE],
        Experiment({'c_A': Measurement(0, times, [1.3, 0.9]),
                    'c_A_repeat': Measurement(0, times, [1.1, 0.7])},
                   args=(2.0,)),
        jac_fun=decay_sensitivity)
    assert estimator.measured_ind == [0, 0]
    # [mol/L]
    model = [decay_point(SEED_RATE, t, 0, (2.0,), {}) for t in times]
    slope = [decay_point(SEED_RATE, t, 0, (2.0,), {}, derivative=1)
             for t in times]  # [mol/L*s]
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True),
        [model[0] - 1.3, model[1] - 0.9, model[0] - 1.1, model[1] - 0.7],
        rtol=ROUNDOFF_RTOL)
    np.testing.assert_allclose(
        estimator.get_gradient([SEED_RATE], out_array=True)[0],
        slope + slope, rtol=ROUNDOFF_RTOL)


def test_string_fields_resolve_through_output_names():
    by_index = ParameterEstimation(decay_model, [SEED_RATE],
                                   build_experiments())
    experiments = {}
    for name, experiment in build_experiments().items():
        experiments[name] = Experiment(
            {meas: Measurement(['cA', 'cB', 'T'][m.field], m.x, m.values,
                               units=m.units)
             for meas, m in experiment.measurements.items()},
            args=experiment.args, kwargs=dict(experiment.kwargs))
    by_name = ParameterEstimation(decay_model, [SEED_RATE], experiments,
                                  output_names=['cA', 'cB', 'T'])
    assert by_name.measured_ind == [0, 1, 2]
    np.testing.assert_array_equal(
        by_name.get_objective([SEED_RATE], out_array=True),
        by_index.get_objective([SEED_RATE], out_array=True))


def test_single_experiment_sequence_names_measurements_by_field():
    experiment = Experiment([Measurement(0, [1.0], [1.2]),
                             Measurement('T', [2.0], [310.0])],
                            args=(2.0,))
    assert experiment.measurement_names == ('field_0', 'T')
    point = Measurement(2, 2.0, 310.0, units='K')  # [s], [K]
    assert point.x.shape == point.values.shape == (1,)
    estimator = ParameterEstimation(decay_model, [SEED_RATE], [experiment],
                                    output_names=['cA', 'cB', 'T'])
    assert estimator.experim_names == ['exp_1']
    assert estimator.name_states == ['field_0', 'T']


# ---------------------------------------------------------------- legacy


def test_legacy_full_grid_matches_experiment_and_closed_form():
    times = np.array([0.0, 0.5, 1.0, 2.0, 4.0])  # [s]
    kwargs = {'adiabatic_rise_k': 40.0}  # [K]
    # [mol/L], [K]
    observed = decay_model([TRUE_RATE], times, 2.0, **kwargs)[:, [0, 2]]
    observed = observed + np.array([[0.01, 0.2], [-0.02, -0.1], [0.0, 0.3],
                                    [0.015, -0.2], [-0.01, 0.1]])  # [mol/L, K]
    legacy = ParameterEstimation(decay_model, [SEED_RATE], times, observed,
                                 measured_ind=(0, 2), args_fun=(2.0,),
                                 kwargs_fun=kwargs,
                                 jac_fun=decay_sensitivity)
    modern = ParameterEstimation(
        decay_model, [SEED_RATE],
        Experiment({'c_A': Measurement(0, times, observed[:, 0]),
                    'T': Measurement(2, times, observed[:, 1])},
                   args=(2.0,), kwargs=kwargs),
        jac_fun=decay_sensitivity)
    assert legacy.measured_ind == (0, 2)
    assert legacy.x_masks == [None]
    np.testing.assert_array_equal(legacy.x_model[0], times)
    # Closed form in the legacy state-major order.
    # [mol/L], [K]
    model = decay_model([SEED_RATE], times, 2.0, **kwargs)[:, [0, 2]]
    residual = (model - observed).T.ravel()  # [mol/L, K]
    sensitivity = np.concatenate(
        (-2.0 * times * np.exp(-SEED_RATE * times),
         40.0 * times * np.exp(-SEED_RATE * times)))  # [mol/L*s, K*s]
    for estimator in (legacy, modern):
        np.testing.assert_array_equal(
            estimator.get_objective([SEED_RATE], out_array=True), residual)
        np.testing.assert_allclose(
            estimator.get_gradient([SEED_RATE], out_array=True)[0],
            sensitivity, rtol=ROUNDOFF_RTOL)
        assert estimator.num_data_total == observed.size
    results = [estimator.optimize_fn(verbose=False)
               for estimator in (legacy, modern)]
    for legacy_value, modern_value in zip(results[0][:2], results[1][:2]):
        np.testing.assert_array_equal(legacy_value, modern_value)
    np.testing.assert_array_equal(results[0][2]['jac'], results[1][2]['jac'])


def test_legacy_replicate_times_keep_grid_and_residual_order():
    # The Ziegler 723 K workshop data repeat 12.5 s and 15 s.
    times = np.array([0., 2.5, 12.5, 12.5, 15., 15.])  # [s]
    # Columns c_A [mol/L] and T [K]; replicate rows differ.
    observed = np.array([[2.0, 300.], [0.6, 333.], [0.01, 340.],
                         [0.02, 339.], [0.0, 341.], [0.005, 340.5]])
    kwargs = {'adiabatic_rise_k': 40.0}  # [K]
    legacy = ParameterEstimation(decay_model, [SEED_RATE], {'723K': times},
                                 {'723K': observed}, measured_ind=[0, 2],
                                 args_fun={'723K': (2.0,)},
                                 kwargs_fun={'723K': kwargs},
                                 jac_fun=decay_sensitivity)
    np.testing.assert_array_equal(legacy.x_model[0], times)
    assert legacy.x_masks == [None]
    # [mol/L], [K]
    model = decay_model([SEED_RATE], times, 2.0, **kwargs)[:, [0, 2]]
    np.testing.assert_array_equal(
        legacy.get_objective([SEED_RATE], out_array=True),
        (model - observed).T.ravel())
    layout = legacy.get_residual_layout()
    assert layout['x'].tolist() == times.tolist() * 2
    assert layout['measurement'].tolist() == ['y_0'] * 6 + ['y_1'] * 6


# Staggered linear GLS fixture: per-run (c_0, c_1) sampling times [s],
# measurement errors [mol/L], initial values [mol/L] and gains [-].
GLS_GRIDS = {'run1': ([0.5, 1.0, 2.0], [1.0, 3.0]),
             'run2': ([0.8, 2.5], [0.8, 1.6, 2.5])}  # [s]
GLS_OFFSETS = {'run1': ([0.1, -0.05, 0.2], [-0.1, 0.15]),
               'run2': ([0.05, -0.2], [0.1, -0.05, 0.2])}  # [mol/L]
GLS_INITIAL = {'run1': 1.0, 'run2': 0.5}  # [mol/L]
GLS_GAIN = {'run1': 2.0, 'run2': 3.0}  # [-]
GLS_RATE = 0.8  # [mol/L/s], synthetic truth
# Power-of-two scale 1/16 keeps the covariance exact; correlation
# -2/sqrt(5) [-]. Rows and columns in (c_0, c_1) order.
GLS_COVARIANCE = np.array([[5.0, -2.0], [-2.0, 1.0]]) * 0.0625  # [(mol/L)**2]
GLS_NUM_OBSERVED = 10  # [-], (3 + 2) + (2 + 3) observations


def linear_model(params, time_s, initial, gain=1.0):
    """Evaluate two outputs linear in a production rate.

    Parameters
    ----------
    params : array_like
        Production rate k [mol/L/s], shape (1,).
    time_s : numpy.ndarray
        Model times [s], shape (n_times,).
    initial : float
        Initial concentration [mol/L].
    gain : float, optional
        Rate multiplier of the second output [-].

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 2): ``initial + k t`` and ``initial + gain k t``
        [mol/L].
    """
    return np.column_stack((initial + params[0] * time_s,
                            initial + gain * params[0] * time_s))


def linear_sensitivity(params, time_s, initial, gain=1.0):
    """Return the state-major rate sensitivities of ``linear_model``.

    Parameters
    ----------
    params, time_s, initial, gain
        As for ``linear_model``.

    Returns
    -------
    numpy.ndarray
        Shape (2 * n_times, 1): ``t`` then ``gain * t`` [s].
    """
    return np.concatenate((time_s, gain * time_s))[:, np.newaxis]


def gls_truth(run, state, time):
    """Return one noise-free output of the GLS fixture.

    Parameters
    ----------
    run : str
        Key of ``GLS_GRIDS``.
    state : int
        Output column, 0 or 1.
    time : float
        Sample time [s].

    Returns
    -------
    float
        Concentration [mol/L].
    """
    gain = GLS_GAIN[run] if state else 1.0  # [-]
    return GLS_INITIAL[run] + gain * GLS_RATE * time


def gls_observations():
    """Return the staggered GLS observations.

    Returns
    -------
    dict
        Run -> list of two arrays [mol/L], one per output, on
        ``GLS_GRIDS``.
    """
    return {run: [np.array([gls_truth(run, state, time) + error
                            for time, error in zip(GLS_GRIDS[run][state],
                                                   GLS_OFFSETS[run][state])])
                  for state in (0, 1)] for run in GLS_GRIDS}


def gls_rows(observations):
    """List the observed sample rows of the GLS fixture.

    Parameters
    ----------
    observations : dict
        Output of ``gls_observations`` [mol/L].

    Returns
    -------
    list of tuple
        ``(data - initial, regressor, precision)`` per observed sample row:
        shifted observations [mol/L] and regressors [s] of the observed
        outputs, and the marginal precision of those outputs
        [(mol/L)**-2] from ``numpy.linalg.inv``.
    """
    rows = []
    for run, grids in GLS_GRIDS.items():
        for time in sorted(set(grids[0]) | set(grids[1])):
            states = [state for state in (0, 1) if time in grids[state]]
            data = np.array([observations[run][state][grids[state].index(time)]
                             for state in states])  # [mol/L]
            regressor = np.array([(GLS_GAIN[run] if state else 1.0) * time
                                  for state in states])  # [s]
            precision = np.linalg.inv(
                GLS_COVARIANCE[np.ix_(states, states)])  # [(mol/L)**-2]
            rows.append((data - GLS_INITIAL[run], regressor, precision))
    return rows


def gls_objective(rows, rate):
    """Return the closed-form GLS objective.

    Parameters
    ----------
    rows : list of tuple
        Output of ``gls_rows``.
    rate : float
        Production rate k [mol/L/s].

    Returns
    -------
    float
        ``1/2 * sum r^T P r`` over observed rows [-].
    """
    return 0.5 * sum((rate * x - y) @ p @ (rate * x - y) for y, x, p in rows)


def gls_experiments(reverse=False):
    """Build the GLS fixture as Experiment objects.

    Parameters
    ----------
    reverse : bool, optional
        Reverse the experiment order; 'run1' always declares 'y_1' before
        'y_0', so the measurement column order follows the experiment order.

    Returns
    -------
    dict
        Run -> Experiment.
    """
    observations = gls_observations()  # [mol/L]
    experiments = {}
    for run in (reversed(list(GLS_GRIDS)) if reverse else GLS_GRIDS):
        order = ('y_1', 'y_0') if run == 'run1' else ('y_0', 'y_1')
        experiments[run] = Experiment(
            {name: Measurement(int(name[-1]),
                               GLS_GRIDS[run][int(name[-1])],
                               observations[run][int(name[-1])])
             for name in order},
            args=(GLS_INITIAL[run],), kwargs={'gain': GLS_GAIN[run]})
    return experiments


def test_legacy_staggered_correlated_gls_matches_experiment_and_closed_form():
    """Named staggered grids, callback args/kwargs and correlated weights.

    The Experiment estimators take the covariance as a labelled DataFrame,
    so reversing the experiments (and hence the measurement column order)
    must not re-weight the fit.
    """
    observations = gls_observations()  # [mol/L]
    legacy = ParameterEstimation(
        linear_model, [1.0],
        {run: [np.array(grid) for grid in GLS_GRIDS[run]]
         for run in GLS_GRIDS},
        observations,
        args_fun={run: (GLS_INITIAL[run],) for run in GLS_GRIDS},
        kwargs_fun={run: {'gain': GLS_GAIN[run]} for run in GLS_GRIDS},
        weight_matrix=GLS_COVARIANCE, jac_fun=linear_sensitivity)
    # Labels in reverse order: the DataFrame is reindexed by name.
    labelled = pd.DataFrame(GLS_COVARIANCE[::-1, ::-1],
                            index=['y_1', 'y_0'],
                            columns=['y_1', 'y_0'])  # [(mol/L)**2]
    modern = [ParameterEstimation(linear_model, [1.0],
                                  gls_experiments(reverse),
                                  weight_matrix=labelled,
                                  jac_fun=linear_sensitivity)
              for reverse in (False, True)]
    assert modern[0].get_residual_layout()['measurement'].iloc[0] == 'y_1'
    assert modern[1].get_residual_layout()['measurement'].iloc[0] == 'y_0'

    np.testing.assert_array_equal(
        legacy.x_masks[0], [[True, False], [True, True], [True, False],
                            [False, True]])
    np.testing.assert_array_equal(
        legacy.x_masks[1], [[True, True], [False, True], [True, True]])

    rows = gls_rows(observations)
    information = sum(x @ p @ x for _, x, p in rows)  # [s**2/(mol/L)**2]
    score = sum(x @ p @ y for y, x, p in rows)  # [s/(mol/L)]
    rate_hat = score / information  # [mol/L/s]
    variance = (2 * gls_objective(rows, rate_hat)
                / (GLS_NUM_OBSERVED - 1))  # [-]

    results = []
    for estimator in (legacy, *modern):
        assert estimator.num_data_total == GLS_NUM_OBSERVED
        assert estimator.get_objective([1.0]) == pytest.approx(
            gls_objective(rows, 1.0), rel=ROUNDOFF_RTOL)
        results.append(estimator.optimize_fn(verbose=False))
    for fitted, covariance, _ in results:
        np.testing.assert_allclose(fitted, [rate_hat], rtol=FIT_RTOL)
        # Covariance depends on the fit only through the SSE, quadratic in
        # the rate error, so FIT_RTOL bounds it as well.
        np.testing.assert_allclose(covariance, [[variance / information]],
                                   rtol=FIT_RTOL)
    np.testing.assert_allclose(results[2][0], results[1][0],
                               rtol=ROUNDOFF_RTOL)


def test_legacy_nan_observation_is_missing():
    times = np.array([0.0, 1.0, 2.0])  # [s]
    observed = np.array([2.0, np.nan, 0.8])  # [mol/L]
    estimator = ParameterEstimation(decay_model, [SEED_RATE], times,
                                    observed, args_fun=(2.0,))
    # [mol/L]
    model = [decay_point(SEED_RATE, t, 0, (2.0,), {}) for t in times]
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True),
        [model[0] - 2.0, 0.0, model[2] - 0.8], rtol=ROUNDOFF_RTOL)
    assert estimator.num_data_total == 2
    np.testing.assert_array_equal(estimator.x_masks[0],
                                  [[True], [False], [True]])


# ------------------------------------------------------------ validation


def measurement_error(match, exception=ValueError, **overrides):
    """Assert that building one measurement fails with a message.

    Parameters
    ----------
    match : str
        Regular expression expected in the message.
    exception : type, optional
        Expected exception class.
    **overrides
        Replacements of the valid arguments field=0, x=[0, 1, 2] [s],
        values=[1, 2, 3] [mol/L].
    """
    arguments = dict(field=0, x=[0.0, 1.0, 2.0], values=[1.0, 2.0, 3.0])
    arguments.update(overrides)
    with pytest.raises(exception, match=match):
        Measurement(arguments.pop('field'), arguments.pop('x'),
                    arguments.pop('values'), **arguments)


@pytest.mark.parametrize('overrides, exception, match', [
    ({'x': [0.0, 2.0, 1.0]}, ValueError,
     r'non-decreasing; x\[2\] = 1\.0 is smaller than x\[1\] = 2\.0'),
    ({'x': [0.0, np.nan, 2.0]}, ValueError, r'finite; x\[1\] = nan'),
    ({'x': [], 'values': []}, ValueError, 'at least one sample'),
    ({'x': [[0.0, 1.0]], 'values': [1.0, 2.0]}, ValueError,
     'one-dimensional'),
    ({'values': [1.0, 2.0]}, ValueError,
     r'got values shape \(2,\) and x shape \(3,\)'),
    ({'values': [1.0, np.inf, 3.0]}, ValueError, r'values\[1\] = inf'),
    ({'values': [np.nan] * 3}, ValueError, 'all NaN'),
    ({'uncertainty': [0.1, 0.0, 0.1]}, ValueError,
     r'positive standard deviation .* uncertainty\[1\] = 0\.0'),
    ({'uncertainty': [0.1, 0.2]}, ValueError, 'one entry per x sample'),
    ({'field': True}, TypeError, 'got a bool'),
    ({'field': -1}, ValueError,
     'non-negative column index of the model output; got -1'),
    ({'field': 1.0}, TypeError, 'got float'),
    ({'units': 3}, TypeError, 'units must be a str or None'),
    ({'field': ''}, ValueError, 'field name must be a non-empty str'),
    ({'x': ['a', 'b', 'c']}, TypeError, 'Measurement x must be numeric'),
    ({'x': [0.0, 1.0j, 2.0]}, TypeError,
     'Measurement x must be real; got complex values'),
    ({'values': np.array([1.0, 2.0, 3.0 + 1.0j])}, TypeError,
     'Measurement values must be real; got complex values'),
    ({'uncertainty': [0.1, 0.1j, 0.1]}, TypeError,
     'Measurement uncertainty must be real; got complex values'),
    ({'uncertainty': 0.1j}, TypeError,
     'Measurement uncertainty must be real; got complex values'),
])
def test_measurement_rejects_invalid_input(overrides, exception, match):
    measurement_error(match, exception, **overrides)


def test_experiment_rejects_invalid_measurement_collections():
    measurement = Measurement(0, [0.0], [1.0])  # [s], [mol/L]
    with pytest.raises(TypeError, match=r"measurements\[1\] must be a "
                       "Measurement; got float"):
        Experiment([measurement, 1.0])
    with pytest.raises(TypeError, match='or a sequence of Measurement '
                       'objects; got int'):
        Experiment(3)
    with pytest.raises(ValueError, match='measurement names must be '
                       'non-empty'):
        Experiment({'': measurement})
    with pytest.raises(ValueError, match="repeat the names \\['field_0'\\]; "
                       "pass a mapping with distinct names"):
        Experiment([measurement, measurement])
    with pytest.raises(ValueError, match='at least one Measurement'):
        Experiment({})
    with pytest.raises(TypeError, match='wrap a single Measurement'):
        Experiment(measurement)
    with pytest.raises(TypeError, match="measurement 'c' must be a "
                       "Measurement; got list"):
        Experiment({'c': [0.0]})
    with pytest.raises(TypeError, match='names must be str; got int'):
        Experiment({0: measurement})
    with pytest.raises(TypeError, match='args must be a tuple or list'):
        Experiment([measurement], args=2.0)
    with pytest.raises(TypeError, match='kwargs must be a mapping'):
        Experiment([measurement], kwargs=[('gain', 1.0)])
    with pytest.raises(TypeError, match='kwargs keys must be str callback '
                       'keyword names; got int 1'):
        Experiment([measurement], kwargs={1: 1.0})
    with pytest.raises(ValueError, match="share x_units; got \\{'a': 's', "
                       "'b': 'min'\\}"):
        Experiment({'a': measurement,
                    'b': Measurement(0, [0.0], [1.0], x_units='min')})


def single(name='c_A', field=0, **options):
    """Return a one-measurement experiment of the decay model.

    Parameters
    ----------
    name : str, optional
        Measurement name.
    field : int or str, optional
        Model-output field.
    **options
        Measurement keywords (units, basis, x_units, uncertainty).

    Returns
    -------
    Experiment
        Two samples at 1 and 2 s [s] with initial A 2 mol/L.
    """
    return Experiment({name: Measurement(field, [1.0, 2.0], [1.0, 0.8],
                                         **options)}, args=(2.0,))


@pytest.mark.parametrize('experiments, options, match', [
    ({'a': single(field='cC')}, {'output_names': ['cA', 'cB', 'T']},
     r"unknown model output 'cC'; available output_names: "
     r"\['cA', 'cB', 'T'\]"),
    ({'a': single(field='cA')}, {}, 'output_names was not given'),
    ({'a': single(field=3)}, {'output_names': ['cA', 'cB', 'T']},
     'selects model output column 3, but output_names declares 3'),
    ({'a': single(units='mol/L'), 'b': single(units='g/L')}, {},
     "Inconsistent units for measurement name 'c_A': measurement 'c_A' of "
     "experiment 'a' declares 'mol/L', but measurement 'c_A' of experiment "
     "'b' declares 'g/L'"),
    ({'a': single(basis='molar'), 'b': single('other', basis='mass')}, {},
     'Inconsistent basis for model output column 0'),
    ({'a': single(), 'b': single(x_units='min')}, {},
     "Inconsistent x_units for all experiments"),
    ({'a': single(), 'b': single(field=1)}, {},
     "'c_A' resolves to model output column 0 in experiment 'a' but to "
     "column 1 in experiment 'b'"),
    ({'a': single(uncertainty=0.1), 'b': single()}, {},
     "Either every Measurement declares an uncertainty or none does"),
    ({'a': single(uncertainty=0.1)}, {'weight_matrix': np.eye(1)},
     'weight_matrix must be None when measurements declare an uncertainty'),
    ({'a': single()}, {'output_names': ['cA', 'cA']},
     r"output_names must be unique; repeated \['cA'\]"),
    ({'a': single()}, {'output_names': []},
     'output_names must name at least one output column'),
    ({'a': single()}, {'output_names': ['cA', '']},
     'output_names entries must be non-empty'),
    ({'': single()}, {}, 'Experiment names must be non-empty'),
    ({'a': single()}, {'weight_matrix': pd.DataFrame(
        np.eye(2), index=['c_A', 'pH'], columns=['c_A', 'pH'])},
     r"labels must be exactly the measurement names; missing=\[\], "
     r"unexpected=\['pH'\]"),
    ({'a': single()}, {'weight_matrix': pd.DataFrame(
        np.eye(1), index=['c_B'], columns=['c_B'])},
     r"missing=\['c_A'\], unexpected=\['c_B'\]"),
    ({'a': single()}, {'weight_matrix': pd.DataFrame(
        np.eye(1), index=['c_A'], columns=['c_B'])},
     r"index and columns must hold the same measurement names; index "
     r"only: \['c_A'\], columns only: \['c_B'\]"),
    ({'a': single()}, {'weight_matrix': pd.DataFrame(
        np.eye(2), index=['c_A', 'c_A'], columns=['c_A', 'c_B'])},
     r"weight_matrix index repeats the labels \['c_A'\]"),
])
def test_experiment_input_rejects_inconsistent_declarations(
        experiments, options, match):
    with pytest.raises(ValueError, match=match):
        ParameterEstimation(decay_model, [SEED_RATE], experiments, **options)


def test_undeclared_units_agree_with_declared_units():
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE],
        {'a': single(units='mol/L', basis='molar'), 'b': single(),
         'c': single(x_units=None)})
    assert estimator.num_data_total == 6


@pytest.mark.parametrize('label', ['y_data', 'measured_ind', 'args_fun',
                                   'kwargs_fun'])
def test_experiment_input_rejects_legacy_arguments(label):
    value = {'y_data': np.ones(2), 'measured_ind': [0], 'args_fun': (2.0,),
             'kwargs_fun': {'gain': 1.0}}[label]
    with pytest.raises(ValueError, match=f'^{label} must be None when x_data '
                       'holds Experiment objects'):
        ParameterEstimation(decay_model, [SEED_RATE], single(),
                            **{label: value})


@pytest.mark.parametrize('x_data', [
    {'a': single(), 'b': np.array([1.0, 2.0])},
    [single(), np.array([1.0, 2.0])]], ids=['mapping', 'list'])
def test_mixed_experiment_collections_are_rejected(x_data):
    with pytest.raises(TypeError, match='mixes Experiment objects with other '
                       'entries'):
        ParameterEstimation(decay_model, [SEED_RATE], x_data,
                            y_data=np.ones(2))


def test_out_of_range_field_fails_at_evaluation_with_identity():
    estimator = ParameterEstimation(decay_model, [SEED_RATE],
                                    {'hot': single('extra', field=3)})
    with pytest.raises(ValueError, match="Measurement 'extra' selects model "
                       r"output column 3, but the model callback returned 3 "
                       r"output column\(s\) for experiment 'hot'"):
        estimator.get_objective([SEED_RATE])


def test_output_names_must_match_callback_columns():
    estimator = ParameterEstimation(decay_model, [SEED_RATE],
                                    {'hot': single(field='cA')},
                                    output_names=['cA', 'cB', 'T', 'pH'])
    with pytest.raises(ValueError, match="returned 3 output columns for "
                       "experiment 'hot', but output_names declares 4"):
        estimator.get_objective([SEED_RATE])


@pytest.mark.parametrize('form', ['experiment', 'measurement'])
def test_multiple_curve_resolution_rejects_experiments(form):
    entry = (single() if form == 'experiment'
             else Measurement(0, [1.0, 2.0], [1.0, 0.8]))  # [s], [mol/L]
    with pytest.raises(TypeError, match='MultipleCurveResolution does not '
                       'accept Experiment or Measurement objects'):
        MultipleCurveResolution(decay_model, [SEED_RATE], {'a': entry},
                                np.ones((2, 3)), measured_ind=[0])


@pytest.mark.parametrize('form', ['bare', 'list', 'tuple', 'mapping',
                                  'mapping_of_lists'])
def test_bare_measurements_must_be_wrapped_in_experiments(form):
    measurement = Measurement(0, [1.0, 2.0], [1.0, 0.8])  # [s], [mol/L]
    x_data = {'bare': measurement, 'list': [measurement],
              'tuple': (measurement,), 'mapping': {'run': measurement},
              'mapping_of_lists': {'run': [measurement]}}[form]
    with pytest.raises(TypeError, match=r"x_data holds Measurement objects; "
                       r"wrap each experiment's measurements in "
                       r"Experiment\(\.\.\.\)"):
        ParameterEstimation(decay_model, [SEED_RATE], x_data)


TIMES_3 = np.array([0.0, 1.0, 2.0])  # [s]


@pytest.mark.parametrize('x_data, y_data, options, exception, match', [
    ({'a': [np.array([0.0, 2.0, 1.0])]}, {'a': [np.ones(3)]}, {},
     ValueError, r"x_data grid 0 of experiment 'a' must be non-decreasing; "
     r"x\[2\] = 1\.0 is smaller than x\[1\] = 2\.0"),
    ({'a': [np.array([0.0, np.nan])]}, {'a': [np.ones(2)]}, {},
     ValueError, "x_data grid 0 of experiment 'a' must be finite"),
    ({'a': [np.ones((2, 2))]}, {'a': [np.ones(2)]}, {},
     ValueError, r"grid 0 of experiment 'a' must be one-dimensional; got "
     r"shape \(2, 2\)"),
    ({'a': [TIMES_3, TIMES_3]}, {'a': [np.ones(3)]}, {},
     ValueError, "Experiment 'a' lists 2 x_data grids, so y_data must list "
     "one observation array per grid; got 1"),
    ({'a': [TIMES_3]}, {'a': [np.ones(2)]}, {},
     ValueError, "Experiment 'a': y_data array 0 has 2 rows, but its x_data "
     "grid has 3 samples"),
    (TIMES_3, np.ones(2), {},
     ValueError, "Experiment 'exp_1': y_data has 2 rows, but x_data has 3 "
     "samples"),
    (TIMES_3, np.ones((3, 1, 1)), {},
     ValueError, r"y_data of experiment 'exp_1' must be one- or "
     r"two-dimensional; got shape \(3, 1, 1\)"),
    (TIMES_3, np.ones(3), {'measured_ind': [0, 2]},
     ValueError, "Experiment 'exp_1' has 1 y_data columns, but measured_ind "
     "lists 2 model outputs"),
    (TIMES_3, np.ones(3), {'measured_ind': 0},
     TypeError, "measured_ind must be a sequence of int model-output "
     "columns or output names; got 0"),
    (TIMES_3, np.ones(3), {'measured_ind': [True]},
     TypeError, "measured_ind entries must be int columns or str output "
     "names; got True"),
    (TIMES_3, np.ones(3), {'weight_matrix': np.eye(2)},
     ValueError, r"one row and column per measurement column \(1: "
     r"\['y_0'\]\); got shape \(2, 2\)"),
    (TIMES_3, np.ones(3), {'measured_ind': [3], 'output_names': ['a', 'b']},
     ValueError, "selects model output column 3, but output_names declares "
     "2 output columns"),
])
def test_legacy_inputs_report_invalid_data(x_data, y_data, options,
                                           exception, match):
    with pytest.raises(exception, match=match):
        ParameterEstimation(decay_model, [SEED_RATE], x_data, y_data,
                            args_fun=[(2.0,)], **options)


@pytest.mark.parametrize('experiments, options, match', [
    ({'a': single()}, {'output_names': 'cA'},
     'output_names must be a sequence of str'),
    ({'a': single()}, {'output_names': ['cA', 2]},
     'output_names entries must be str; got 2'),
    ({1: single()}, {}, 'Experiment names must be str; got int 1'),
    ({'a': single()}, {'weight_matrix': np.eye(1)},
     r"weight_matrix must be a pandas DataFrame whose index and columns are "
     r"the measurement names \['c_A'\]"),
])
def test_experiment_input_rejects_invalid_types(experiments, options, match):
    with pytest.raises(TypeError, match=match):
        ParameterEstimation(decay_model, [SEED_RATE], experiments, **options)


def test_x_units_conflict_names_the_declaring_measurement():
    first = Experiment({'undeclared': Measurement(0, [1.0], [1.0],
                                                  x_units=None),
                        'declared': Measurement(1, [1.0], [1.0])},
                       args=(2.0,))  # [s], [mol/L]
    with pytest.raises(ValueError, match="Inconsistent x_units for all "
                       "experiments, because one model callback receives "
                       "every grid: measurement 'declared' of experiment "
                       "'a' declares 's', but measurement 'c_A' of "
                       "experiment 'b' declares 'min'"):
        ParameterEstimation(decay_model, [SEED_RATE],
                            {'a': first, 'b': single(x_units='min')})


def test_nested_observations_reject_output_names_and_layout():
    times = {'run': {'spectra': np.array([0.0, 1.0]),
                     'non_spectra': np.array([0.5])}}  # [s]
    observations = {'run': {'spectra': np.ones((2, 2)),  # [-]
                            'non_spectra': np.ones(1)}}  # [mol/L]
    with pytest.raises(ValueError, match='output_names is not supported '
                       'with nested state observation dictionaries'):
        ParameterEstimation(decay_model, [SEED_RATE], times, observations,
                            output_names=['cA'])
    estimator = MultipleCurveResolution(
        decay_model, [SEED_RATE], times, observations,
        measured_ind={'spectra': [0], 'non_spectra': [0]},
        name_states=['c_A'])
    with pytest.raises(NotImplementedError, match='get_residual_layout is '
                       'not available for nested state observation'):
        estimator.get_residual_layout()


def test_uncertainty_scaling_divides_without_overflow():
    sigma = 1e-310  # [mol/L], subnormal: 1 / sigma overflows to inf
    offset = 2.0 ** -40  # [mol/L], exactly representable data offset
    experiment = Experiment([Measurement(0, [0.0], [2.0 + offset],
                                         uncertainty=sigma)],
                            args=(2.0,))  # c_A(0) = 2 mol/L exactly
    estimator = ParameterEstimation(decay_model, [SEED_RATE], experiment)
    residual = estimator.get_objective([SEED_RATE], out_array=True)  # [-]
    assert np.all(np.isfinite(residual))
    np.testing.assert_array_equal(residual, [-offset / sigma])


# ------------------------------------------------- legacy compatibility


def arrhenius_rate(params, temp_k):
    """Evaluate an Arrhenius rate constant.

    Parameters
    ----------
    params : array_like
        Pre-exponential factor A [1/s] and activation temperature
        Ea/R [K], shape (2,).
    temp_k : numpy.ndarray
        Temperatures [K], any order, shape (n,).

    Returns
    -------
    numpy.ndarray
        Rate constants [1/s], shape (n,).
    """
    return params[0] * np.exp(-params[1] / temp_k)


def test_legacy_unsorted_grid_is_passed_through_unchanged():
    temps = np.array([350.0, 320.0, 340.0, 330.0])  # [K], not sorted
    rates = np.array([3.0, 1.0, 2.0, 1.5])  # [1/s], synthetic observations
    params = [1e3, 2e3]  # [1/s], [K]
    estimator = ParameterEstimation(arrhenius_rate, params, temps, rates)
    assert estimator.x_model[0] is temps
    assert estimator.x_masks == [None]
    expected = [1e3 * np.exp(-2e3 / temp) - rate
                for temp, rate in zip(temps, rates)]  # [1/s]
    np.testing.assert_allclose(estimator.get_objective(params,
                                                       out_array=True),
                               expected, rtol=ROUNDOFF_RTOL)
    assert estimator.get_residual_layout()['x'].tolist() == temps.tolist()


def design_response(params, design):
    """Evaluate a linear regression on a design matrix.

    Parameters
    ----------
    params : array_like
        Intercept [mol/L] and slope [mol/L/s], shape (2,).
    design : numpy.ndarray
        Design matrix, shape (n, 2): ones [-] and times [s].

    Returns
    -------
    numpy.ndarray
        Concentrations [mol/L], shape (n,).
    """
    return design @ np.asarray(params)


def column_response(params, time_column):
    """Evaluate a line on a column-vector grid.

    Parameters
    ----------
    params : array_like
        Slope [mol/L/s] and intercept [mol/L], shape (2,).
    time_column : numpy.ndarray
        Times [s], shape (n, 1).

    Returns
    -------
    numpy.ndarray
        Concentrations [mol/L], shape (n,).
    """
    return params[0] * time_column[:, 0] + params[1]


def table_lookup(params, index, table):
    """Scale tabulated values selected by an integer grid.

    Parameters
    ----------
    params : array_like
        Scale factor [-], shape (1,).
    index : numpy.ndarray
        Integer row indices into ``table``, shape (n,).
    table : list of float
        Tabulated concentrations [mol/L].

    Returns
    -------
    numpy.ndarray
        Scaled concentrations [mol/L], shape (n,).
    """
    return params[0] * np.asarray(table)[index]


def test_legacy_multidimensional_and_integer_grids_keep_their_meaning():
    design = np.array([[1.0, 0.5], [1.0, 2.0], [1.0, 1.0]])  # [-], [s]
    observed = np.array([1.6, 4.9, 3.1])  # [mol/L]
    regression = ParameterEstimation(design_response, [1.0, 2.0], design,
                                     observed)
    assert regression.x_model[0] is design
    # 1 + 2 t at t = 0.5, 2, 1 s minus the observations [mol/L].
    np.testing.assert_allclose(
        regression.get_objective([1.0, 2.0], out_array=True),
        [2.0 - 1.6, 5.0 - 4.9, 3.0 - 3.1], rtol=ROUNDOFF_RTOL)

    column = np.array([[0.0], [1.0], [3.0]])  # [s], column vector
    line = ParameterEstimation(column_response, [2.0, 1.0], column,
                               np.array([1.0, 3.5, 6.5]))  # [mol/L]
    assert line.x_model[0] is column
    np.testing.assert_allclose(line.get_objective([2.0, 1.0],
                                                  out_array=True),
                               [0.0, -0.5, 0.5], rtol=ROUNDOFF_RTOL)

    index = np.array([2, 0, 1])  # [-], integer rows used by the callback
    table = [1.0, 2.0, 4.0]  # [mol/L]
    arguments = (table,)  # positional callback arguments
    lookup = ParameterEstimation(table_lookup, [2.0], index,
                                 np.array([8.5, 2.0, 4.0]),  # [mol/L]
                                 args_fun=arguments)
    assert lookup.x_model[0] is index
    assert lookup.args_fun[0] is arguments
    np.testing.assert_allclose(lookup.get_objective([2.0], out_array=True),
                               [-0.5, 0.0, 0.0], rtol=ROUNDOFF_RTOL)


def scaled_float32(params, time_s):
    """Return a single-precision linear response.

    Parameters
    ----------
    params : array_like
        Slope [mol/L/s], shape (1,).
    time_s : numpy.ndarray
        Times [s], float32, shape (n,).

    Returns
    -------
    numpy.ndarray
        Concentrations [mol/L], float32, shape (n,).
    """
    return (params[0] * time_s).astype(np.float32)


def test_legacy_dtypes_and_callback_containers_are_preserved():
    times = np.array([0.0, 1.0, 3.0], dtype=np.float32)  # [s]
    observed = np.array([0.5, 2.0, 6.0], dtype=np.float32)  # [mol/L]
    estimator = ParameterEstimation(scaled_float32, [2.0], times, observed)
    assert estimator.x_model[0] is times
    assert estimator.y_data[0].dtype == np.float32
    np.testing.assert_array_equal(estimator.y_data[0][:, 0], observed)
    np.testing.assert_array_equal(
        estimator.get_objective([2.0], out_array=True),
        np.array([-0.5, 0.0, 0.0], dtype=np.float32))

    arguments = np.array([2.0])  # [mol/L], array argument container
    keywords = {'adiabatic_rise_k': 10.0}  # [K]
    legacy = ParameterEstimation(decay_model, [SEED_RATE], TIMES_3,
                                 np.ones(3), args_fun=arguments,
                                 kwargs_fun=keywords)
    assert legacy.args_fun[0] is arguments
    assert legacy.kwargs_fun[0] is keywords

    grids = [np.array([0, 2]), np.array([1, 2])]  # [s], integer grids
    staggered = ParameterEstimation(decay_model, [SEED_RATE], {'a': grids},
                                    {'a': [np.ones(2), np.ones(2)]},
                                    args_fun=[(2.0,)])
    assert staggered.x_model[0].dtype == np.array([0, 1, 2]).dtype
    np.testing.assert_array_equal(staggered.x_model[0], [0, 1, 2])
    assert staggered.y_data[0].dtype == np.float64


def test_legacy_negative_and_named_fields_select_model_outputs():
    times = np.array([0.0, 1.0, 2.0])  # [s]
    kwargs = {'adiabatic_rise_k': 40.0}  # [K]
    temps = [decay_point(TRUE_RATE, t, 2, (2.0,), kwargs)
             for t in times]  # [K]
    observed = np.column_stack((np.full(3, 1.0), temps))  # [mol/L], [K]
    # [mol/L], [mol/L], [K]
    model = decay_model([SEED_RATE], times, 2.0, **kwargs)
    expected = np.concatenate((model[:, 0] - 1.0,
                               model[:, 2] - temps))  # [mol/L], [K]
    negative = ParameterEstimation(decay_model, [SEED_RATE], times, observed,
                                   measured_ind=[0, -1], args_fun=(2.0,),
                                   kwargs_fun=kwargs)
    named_fields = ['cA', 'T']
    named = ParameterEstimation(decay_model, [SEED_RATE], times, observed,
                                measured_ind=named_fields, args_fun=(2.0,),
                                kwargs_fun=kwargs,
                                output_names=['cA', 'cB', 'T'])
    assert named.measured_ind is named_fields
    assert named.name_states == ['cA', 'T']
    for estimator in (negative, named):
        np.testing.assert_allclose(
            estimator.get_objective([SEED_RATE], out_array=True), expected,
            rtol=ROUNDOFF_RTOL)
    with pytest.raises(ValueError, match="names model output 'cA', but "
                       "output_names was not given"):
        ParameterEstimation(decay_model, [SEED_RATE], times, observed,
                            measured_ind=named_fields, args_fun=(2.0,))
    with pytest.raises(ValueError, match="Measurement 'y_1' selects model "
                       r"output column -4, but the model callback returned "
                       r"3 output column\(s\)"):
        ParameterEstimation(decay_model, [SEED_RATE], times, observed,
                            measured_ind=[0, -4], args_fun=(2.0,)
                            ).get_objective([SEED_RATE])


def scalar_decay(params, time_s, initial_mol_l):
    """Return c_A of first-order decay as a one-dimensional output.

    Parameters
    ----------
    params : array_like
        Rate constant k [1/s], shape (1,).
    time_s : numpy.ndarray
        Times [s], shape (n,).
    initial_mol_l : float
        Initial concentration [mol/L].

    Returns
    -------
    numpy.ndarray
        Concentrations [mol/L], shape (n,).
    """
    return initial_mol_l * np.exp(-params[0] * time_s)


def scalar_decay_sensitivity(params, time_s, initial_mol_l):
    """Return the rate sensitivity of ``scalar_decay``.

    Parameters
    ----------
    params, time_s, initial_mol_l
        As for ``scalar_decay``.

    Returns
    -------
    numpy.ndarray
        Shape (n, 1): dc_A/dk [mol/L*s].
    """
    return (-initial_mol_l * time_s
            * np.exp(-params[0] * time_s))[:, np.newaxis]


def test_legacy_one_dimensional_output_ignores_single_field():
    observed = np.array([2.0, 1.4, 1.0])  # [mol/L]
    estimator = ParameterEstimation(scalar_decay, [SEED_RATE], TIMES_3,
                                    observed, measured_ind=[2],
                                    args_fun=(2.0,),
                                    jac_fun=scalar_decay_sensitivity)
    # [mol/L]
    model = [decay_point(SEED_RATE, t, 0, (2.0,), {}) for t in TIMES_3]
    slope = [decay_point(SEED_RATE, t, 0, (2.0,), {}, derivative=1)
             for t in TIMES_3]  # [mol/L*s]
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True),
        np.array(model) - observed, rtol=ROUNDOFF_RTOL)
    np.testing.assert_allclose(
        estimator.get_gradient([SEED_RATE], out_array=True)[0], slope,
        rtol=ROUNDOFF_RTOL)


def test_caller_measured_ind_changes_after_construction_are_ignored():
    times = np.array([0.0, 1.0, 2.0])  # [s]
    kwargs = {'adiabatic_rise_k': 40.0}  # [K]
    observed = np.column_stack((np.ones(3), np.full(3, 310.0)))  # [mol/L], [K]
    measured = [0, 2]
    estimator = ParameterEstimation(decay_model, [SEED_RATE], times, observed,
                                    measured_ind=measured, args_fun=(2.0,),
                                    kwargs_fun=kwargs,
                                    jac_fun=decay_sensitivity)
    measured[1] = 1  # caller reuses its list for another estimator
    # [mol/L], [mol/L], [K]
    model = decay_model([SEED_RATE], times, 2.0, **kwargs)
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True),
        np.concatenate((model[:, 0] - 1.0, model[:, 2] - 310.0)),
        rtol=ROUNDOFF_RTOL)
    # The Jacobian is the derivative of those residuals: c_A then T.
    slope = np.concatenate([[decay_point(SEED_RATE, t, field, (2.0,), kwargs,
                                         derivative=1) for t in times]
                            for field in (0, 2)])  # [mol/L*s], [K*s]
    np.testing.assert_allclose(
        estimator.get_gradient([SEED_RATE], out_array=True)[0], slope,
        rtol=ROUNDOFF_RTOL)


def misshaped_output(params, time_s, shape):
    """Return a constant output of a requested shape.

    Parameters
    ----------
    params : array_like
        Unused rate [1/s].
    time_s : numpy.ndarray
        Unused times [s].
    shape : tuple of int
        Output shape.

    Returns
    -------
    numpy.ndarray
        Ones [mol/L] of ``shape``.
    """
    return np.ones(shape)


@pytest.mark.parametrize('shape', [(2, 1, 2), (1, 1), (3,), ()])
def test_callback_output_shape_is_checked_with_identity(shape):
    experiment = Experiment([Measurement(0, [1.0, 2.0], [1.0, 1.0])],
                            args=(shape,))  # [s], [mol/L]
    estimator = ParameterEstimation(misshaped_output, [SEED_RATE],
                                    {'run': experiment})
    with pytest.raises(ValueError, match=(
            "returned an output of shape " + re.escape(str(shape))
            + " for experiment 'run'; expected a one- or two-dimensional "
            "array with 2 rows")):
        estimator.get_objective([SEED_RATE])


def constant_states(params, time_s, shape, reord_sens=False):
    """Return constant curve-resolution states of a requested shape.

    Parameters
    ----------
    params : array_like
        Unused rate [1/s].
    time_s : numpy.ndarray
        Unused times [s].
    shape : tuple of int
        Output shape.
    reord_sens : bool, optional
        Unused curve-resolution flag.

    Returns
    -------
    numpy.ndarray
        Ones [mol/L] of ``shape``.
    """
    return np.ones(shape)


@pytest.mark.parametrize('shape', [(2, 2), (3,)])
def test_curve_resolution_requires_two_dimensional_rows(shape):
    times = np.array([0.0, 1.0, 2.0])  # [s]
    estimator = MultipleCurveResolution(
        constant_states, [SEED_RATE], times, np.ones((3, 2)),  # [-]
        measured_ind=[0], args_fun=(shape,), name_states=['c_A'])
    with pytest.raises(ValueError, match=(
            "shape " + re.escape(str(shape)) + " for experiment 'exp_1'; "
            "expected a two-dimensional array with 3 rows")):
        estimator.get_objective([SEED_RATE])


def test_labelled_weights_read_lower_triangle_in_caller_order():
    """Only the lower triangle in the caller's label order is read.

    The upper entry 0 differs from the lower entry 1, so reading the
    triangle after reordering by name would change the weighting.
    """
    times = [1.0, 2.0]  # [s]
    first = Measurement(0, times, [1.2, 0.7])  # [mol/L]
    second = Measurement(1, times, [0.9, 1.3])  # [mol/L]
    # Lower triangle in (a, b) order: covariance [[4, 1], [1, 9]]
    # [(mol/L)**2]; columns listed in another order on purpose.
    labelled = pd.DataFrame([[0.0, 4.0], [9.0, 1.0]], index=['a', 'b'],
                            columns=['b', 'a'])  # [(mol/L)**2]
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])  # [(mol/L)**2]
    model = decay_model([SEED_RATE], np.array(times), 2.0)[:, :2]
    data = np.array([[1.2, 0.9], [0.7, 1.3]])  # [mol/L]
    rows = model - data  # [mol/L], one row per time in (a, b) order
    expected = 0.5 * sum(row @ np.linalg.inv(covariance) @ row
                         for row in rows)  # [-]
    objectives = []
    for measurements in ({'a': first, 'b': second},
                         {'b': second, 'a': first}):
        estimator = ParameterEstimation(
            decay_model, [SEED_RATE],
            Experiment(measurements, args=(2.0,)), weight_matrix=labelled)
        objectives.append(estimator.get_objective([SEED_RATE]))
    assert objectives[0] == pytest.approx(expected, rel=ROUNDOFF_RTOL)
    assert objectives[1] == pytest.approx(expected, rel=ROUNDOFF_RTOL)
    with pytest.raises(TypeError, match='weight_matrix must hold real '
                       'covariances; got complex values'):
        ParameterEstimation(
            decay_model, [SEED_RATE],
            Experiment({'a': first, 'b': second}, args=(2.0,)),
            weight_matrix=labelled.astype(complex))


def constant_int8(params, time_s, level):
    """Return a constant single-byte integer output.

    Parameters
    ----------
    params : array_like
        Unused rate [1/s].
    time_s : numpy.ndarray
        Model times [s], shape (n,).
    level : int
        Output value [count].

    Returns
    -------
    numpy.ndarray
        ``level`` as int8 [count], shape (n,).
    """
    return np.full(len(time_s), level, dtype=np.int8)


def test_legacy_staggered_layout_keeps_float64_and_ndarray_masks():
    # int8 residual 100 - (-100) = 200 overflows to -56 in int8 arithmetic;
    # float64 observations, as in earlier releases, keep it exact.
    grids = [np.array([0, 1])]  # [s]
    observed = [np.array([-100, -100], dtype=np.int8)]  # [count]
    estimator = ParameterEstimation(constant_int8, [SEED_RATE], {'a': grids},
                                    {'a': observed}, args_fun=[(100,)])
    assert estimator.y_data[0].dtype == np.float64
    assert isinstance(estimator.x_masks[0], np.ndarray)
    assert estimator.x_masks[0].all()
    np.testing.assert_array_equal(
        estimator.get_objective([SEED_RATE], out_array=True), [200.0, 200.0])

    single = np.array([0.1, 0.7], dtype=np.float32)  # [mol/L]
    staggered = ParameterEstimation(
        decay_model, [SEED_RATE], {'a': [np.array([1.0, 2.0])]},
        {'a': [single]}, args_fun=[(2.0,)])
    assert staggered.y_data[0].dtype == np.float64
    np.testing.assert_array_equal(staggered.y_data[0][:, 0],
                                  single.astype(np.float64))


@pytest.mark.parametrize('grid, values, exception, match', [
    (np.array([2, 1], dtype=np.uint8), np.ones(2), ValueError,
     r"must be non-decreasing; x\[1\] = 1 is smaller than x\[0\] = 2"),
    (np.array([0.0, 1.0]), np.array([1.0, np.inf]), ValueError,
     r"y_data column 0 of experiment 'a': values must be finite or NaN "
     r"\(missing\); values\[1\] = inf"),
    (np.array([0.0, 1.0]), np.full(2, np.nan), ValueError,
     "y_data column 0 of experiment 'a': values are all NaN; a measurement "
     "needs at least one observation"),
    (np.array([0.0, 1.0]), np.array([1.0, 1.0j]), TypeError,
     "y_data column 0 of experiment 'a': values must be real; got complex "
     "values"),
])
def test_legacy_staggered_observations_are_validated(grid, values, exception,
                                                     match):
    with pytest.raises(exception, match=match):
        ParameterEstimation(decay_model, [SEED_RATE], {'a': [grid]},
                            {'a': [values]}, args_fun=[(2.0,)])


def test_legacy_shared_grid_observations_are_validated():
    with pytest.raises(ValueError, match="y_data column 1 of experiment "
                       "'exp_1': values are all NaN"):
        ParameterEstimation(decay_model, [SEED_RATE], TIMES_3,
                            np.column_stack((np.ones(3), np.full(3, np.nan))),
                            measured_ind=[0, 1], args_fun=(2.0,))
    with pytest.raises(ValueError, match=r"values\[2\] = -inf"):
        ParameterEstimation(decay_model, [SEED_RATE], TIMES_3,
                            np.array([1.0, 0.9, -np.inf]), args_fun=(2.0,))
    with pytest.raises(TypeError, match='values must be real'):
        ParameterEstimation(decay_model, [SEED_RATE], TIMES_3,
                            np.array([1.0, 0.9, 0.5 + 0.1j]),
                            args_fun=(2.0,))
    with pytest.raises(ValueError, match="y_data column 0 of experiment "
                       "'exp_1' has no observations"):
        ParameterEstimation(decay_model, [SEED_RATE], np.array([]),
                            np.array([]), args_fun=(2.0,))


def test_legacy_staggered_empty_grid_leaves_state_unmeasured():
    times = np.array([0.0, 1.0, 2.0])  # [s]
    observed = np.array([2.0, 1.5, 1.1])  # [mol/L]
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE], {'a': [times, np.array([])]},
        {'a': [observed, np.array([])]}, measured_ind=[0, 1],
        args_fun=[(2.0,)])
    np.testing.assert_array_equal(estimator.x_model[0], times)
    np.testing.assert_array_equal(
        estimator.x_masks[0], [[True, False], [True, False], [True, False]])
    assert estimator.num_data_total == 3
    model = [decay_point(SEED_RATE, t, 0, (2.0,), {})
             for t in times]  # [mol/L]
    np.testing.assert_allclose(
        estimator.get_objective([SEED_RATE], out_array=True),
        np.concatenate((np.array(model) - observed, np.zeros(3))),
        rtol=ROUNDOFF_RTOL, atol=0.0)


def test_legacy_numeric_object_observations_mark_missing_values():
    """A pandas nullable export (Float64 and Int64 columns) is numeric."""
    frame = pd.DataFrame({'a': pd.array([2.0, None, 6.0], dtype='Float64'),
                          'b': pd.array([1, 2, 3], dtype='Int64')})
    observed = frame.to_numpy(na_value=np.nan)  # [mol/L], object dtype
    assert observed.dtype == object
    estimator = ParameterEstimation(
        linear_model, [1.0], np.array([1.0, 2.0, 3.0]), observed,
        args_fun=(0.0,), kwargs_fun={'gain': 1.0})  # [s], [mol/L], [-]
    np.testing.assert_array_equal(
        estimator.x_masks[0], [[True, True], [False, True], [True, True]])
    assert estimator.num_data_total == 5
    # Model k t = t at k = 1 mol/L/s: residuals (1 - 2, 3 - 6) for 'a'
    # and zeros for 'b', so 1/2 * (1 + 9) = 5 [(mol/L)**2].
    assert estimator.get_objective([1.0]) == pytest.approx(5.0,
                                                           rel=ROUNDOFF_RTOL)


@pytest.mark.parametrize('values, exception, match', [
    (np.array([1.0, np.inf, 2.0], dtype=object), ValueError,
     r"values\[1\] = inf"),
    (np.array([np.nan, np.nan, np.nan], dtype=object), ValueError,
     'values are all NaN'),
    (np.array(['1.0', '2.0', '3.0'], dtype=object), TypeError,
     "values must be real numbers; got '1.0'"),
    (np.array([1.0, pd.NA, 2.0], dtype=object), TypeError,
     'values must be real numbers; got <NA>'),
    (np.array([1.0, 2.0j, 3.0], dtype=object), TypeError,
     'values must be real numbers; got 2j'),
    (np.array(['a', 'b', 'c']), TypeError,
     'values must be real numbers; got dtype <U1'),
])
def test_legacy_object_observations_are_validated(values, exception, match):
    for x_data, y_data in ((TIMES_3, values), ({'a': [TIMES_3]},
                                                {'a': [values]})):
        with pytest.raises(exception, match=match):
            ParameterEstimation(decay_model, [SEED_RATE], x_data, y_data,
                                args_fun=[(2.0,)])


def test_unobserved_raw_residuals_are_exactly_zero_for_float32_data():
    observed = np.array([2.0, np.nan, 1.0], dtype=np.float32)  # [mol/L]
    estimator = ParameterEstimation(decay_model, [SEED_RATE], TIMES_3,
                                    observed, args_fun=(2.0,))
    estimator.get_objective([SEED_RATE])
    # The model value 2 exp(-0.3) [mol/L] is not a float32 number, so
    # writing it into float32 observations would leave a rounding residue.
    assert estimator.resid_runs[0][1, 0] == 0.0
    assert estimator.objfun_iter[-1] == pytest.approx(
        (decay_point(SEED_RATE, 0.0, 0, (2.0,), {}) - 2.0) ** 2
        + (decay_point(SEED_RATE, 2.0, 0, (2.0,), {}) - 1.0) ** 2,
        rel=ROUNDOFF_RTOL)


def test_legacy_wide_integer_staggered_grid_keeps_all_observations():
    # [s]; the adjacent step 200 overflows int8, so np.diff would wrap it
    # to -56 and wrongly report a decrease.
    grid = np.array([-100, 100], dtype=np.int8)
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE], {'a': [grid, np.array([0], dtype=np.int8)]},
        {'a': [np.ones(2), np.ones(1)]}, args_fun=[(2.0,)])
    np.testing.assert_array_equal(estimator.x_model[0], [-100, 0, 100])
    assert estimator.num_data_total == 3
    np.testing.assert_array_equal(
        estimator.x_masks[0], [[True, False], [False, True], [True, False]])


@pytest.mark.parametrize('shape', [(1, 3), (3, 1)])
def test_legacy_staggered_vector_shaped_grids_are_flattened(shape):
    flat = [np.array([0.0, 1.0, 2.0]), np.array([1.0])]  # [s]
    observations = [np.array([2.0, 1.5, 1.1]), np.array([0.6])]  # [mol/L]
    reference = ParameterEstimation(decay_model, [SEED_RATE], {'a': flat},
                                    {'a': observations}, args_fun=[(2.0,)],
                                    measured_ind=[0, 1])
    shaped = ParameterEstimation(
        decay_model, [SEED_RATE], {'a': [flat[0].reshape(shape), flat[1]]},
        {'a': observations}, args_fun=[(2.0,)], measured_ind=[0, 1])
    np.testing.assert_array_equal(shaped.x_model[0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(shaped.x_masks[0], reference.x_masks[0])
    np.testing.assert_array_equal(
        shaped.get_objective([SEED_RATE], out_array=True),
        reference.get_objective([SEED_RATE], out_array=True))


def test_residual_layout_reports_selected_columns_after_evaluation():
    estimator = ParameterEstimation(scalar_decay, [SEED_RATE], TIMES_3,
                                    np.array([2.0, 1.4, 1.0]),  # [mol/L]
                                    measured_ind=[2], args_fun=(2.0,))
    assert estimator.get_residual_layout()['field'].tolist() == [2, 2, 2]
    estimator.get_objective([SEED_RATE])
    assert estimator.get_residual_layout()['field'].tolist() == [0, 0, 0]

    negative = ParameterEstimation(decay_model, [SEED_RATE], TIMES_3,
                                   np.full(3, 300.0),  # [K]
                                   measured_ind=[-1], args_fun=(2.0,))
    assert negative.get_residual_layout()['field'].tolist() == [-1] * 3
    negative.get_objective([SEED_RATE])
    assert negative.get_residual_layout()['field'].tolist() == [2] * 3


# ---------------------------------------------------------- caller data


def test_measurements_copy_and_lock_caller_arrays():
    times = np.array([1.0, 2.0])  # [s]
    values = np.array([1.0, 0.8])  # [mol/L]
    std = np.array([0.1, 0.1])  # [mol/L]
    measurement = Measurement(0, times, values, uncertainty=std)
    estimator = ParameterEstimation(
        decay_model, [SEED_RATE], Experiment([measurement], args=(2.0,)))
    before = estimator.get_objective([SEED_RATE], out_array=True)
    times[:] = [5.0, 6.0]
    values[:] = 100.0
    std[:] = 1.0
    np.testing.assert_array_equal(measurement.x, [1.0, 2.0])
    np.testing.assert_array_equal(measurement.values, [1.0, 0.8])
    np.testing.assert_array_equal(measurement.uncertainty, [0.1, 0.1])
    np.testing.assert_array_equal(
        estimator.get_objective([SEED_RATE], out_array=True), before)
    for array in (measurement.x, measurement.values, measurement.uncertainty):
        with pytest.raises(ValueError, match='read-only'):
            array[0] = 0.0
        with pytest.raises(ValueError, match='cannot set WRITEABLE flag'):
            array.flags.writeable = True
    np.testing.assert_array_equal(measurement.x, [1.0, 2.0])
    np.testing.assert_array_equal(measurement.values, [1.0, 0.8])


def test_experiment_copies_callback_keywords():
    kwargs = {'product_mol_l': 0.5}  # [mol/L]
    experiment = Experiment([Measurement(1, [1.0], [1.0])], args=[2.0],
                            kwargs=kwargs)
    kwargs['product_mol_l'] = 9.0
    assert experiment.kwargs == {'product_mol_l': 0.5}
    assert experiment.args == (2.0,)
    extra = Measurement(0, [0.0], [1.0])  # [s], [mol/L]
    with pytest.raises(TypeError, match='does not support item assignment'):
        experiment.kwargs['product_mol_l'] = 1.0
    with pytest.raises(TypeError, match='does not support item assignment'):
        experiment.measurements['new'] = extra


def test_repeated_fits_leave_caller_and_measurement_data_unchanged():
    experiments = build_experiments()
    snapshot = {(name, meas): (m.x.copy(), m.values.copy())
                for name, experiment in experiments.items()
                for meas, m in experiment.measurements.items()}
    estimator = ParameterEstimation(decay_model, [SEED_RATE], experiments,
                                    jac_fun=decay_sensitivity)
    first = estimator.optimize_fn(verbose=False)[0]
    second = estimator.optimize_fn(verbose=False)[0]
    np.testing.assert_allclose(second, first, rtol=FIT_RTOL)
    for (name, meas), (x, values) in snapshot.items():
        measurement = experiments[name].measurements[meas]
        np.testing.assert_array_equal(measurement.x, x)
        np.testing.assert_array_equal(measurement.values, values)
    assert dict(experiments['hot'].kwargs) == FIXTURE['hot']['kwargs']

    times = np.array([0.0, 1.0, 2.0])  # [s]
    observed = np.array([2.0, np.nan, 0.8])  # [mol/L]
    legacy = ParameterEstimation(decay_model, [SEED_RATE], times, observed,
                                 args_fun=(2.0,))
    legacy.optimize_fn(verbose=False)
    legacy.optimize_fn(verbose=False)
    np.testing.assert_array_equal(times, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(observed, [2.0, np.nan, 0.8])
