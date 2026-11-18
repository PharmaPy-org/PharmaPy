"""Seeded batch cooling simulation of the caffeine case on both batch crystallizers.

Verification of the simulation path, not a calibration: the kinetic constants
are the starting values of the dataset description. The legacy ``BatchCryst``
and the refactored ``BatchCrystallizer`` solve ``high_seed_slow_cool`` with CVode
at measured sample times. Checked: one row per requested time, the solute and the
solvent stay in the vessel, the two models agree, and the predictions respond to
the kinetic parameters and repeat exactly. Assimulo lane; about ten solves of two
to three seconds.
"""
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from examples.pat import batch_cooling
from PharmaPy import SlurryMeasurements as measurements

CASE = Path(batch_cooling.__file__).parent / "caffeine"
RUN = "high_seed_slow_cool"
SPECTRUM_STRIDE = 10  # every tenth spectrum: 52 unevenly spaced times over 9.2 h
# Measured on this case: the solute in the vessel drifts by 4e-8 of the charge in
# the legacy model and 3e-7 in the refactored one; about three times that is allowed.
BALANCE_TOLERANCE = 1e-6  # [-]
# Measured on this case: the predictions of the two models differ by up to 0.9 %
# (count signal), 0.7 % for the dissolved ratio. The legacy unit treats the liquid
# density as constant, a known limitation. Twice the measured difference is allowed.
PARITY_TOLERANCE = 0.02  # [-]


@pytest.fixture(scope="module")
def case():
    inputs, kinetics = batch_cooling.load_case(CASE, RUN, "common_slope")
    return inputs, kinetics, batch_cooling.measurement_times(CASE, RUN, "ir")[::SPECTRUM_STRIDE]


@pytest.fixture(scope="module")
def states(case):
    inputs, kinetics, times = case
    return {model: batch_cooling.simulate(inputs, kinetics, times, model) for model in batch_cooling.MODELS}


@pytest.mark.parametrize("model", batch_cooling.MODELS)
def test_model_returns_the_requested_rows_and_keeps_the_solute_and_the_solvent(case, states, model):
    inputs, _, times = case
    state = states[model]
    charged = inputs.solute_mass + inputs.seed_mass  # [kg]

    np.testing.assert_array_equal(state.time, times)
    assert state.moments.shape == (times.size, 4)
    np.testing.assert_allclose(measurements.total_solute(state, inputs.solute_density, inputs.shape_factor), charged,
                               rtol=BALANCE_TOLERANCE)
    np.testing.assert_allclose(state.solvent_mass, inputs.solvent_mass, rtol=BALANCE_TOLERANCE)
    # 21 s after seeding the seed has grown by 2 % and the liquid is still the charge.
    ratio = measurements.dissolved_ratio(state)
    assert ratio[0] == pytest.approx(inputs.solute_mass / inputs.solvent_mass, rel=2e-3)
    assert measurements.solid_mass(state, inputs.solute_density, inputs.shape_factor)[0] == pytest.approx(
        inputs.seed_mass, rel=0.05)
    # Dissolution is inactive, so the dissolved solute cannot rise; the legacy
    # states wobble by 5e-8 kg/kg at the end, within the solver tolerance.
    assert np.all(np.diff(ratio) < 1e-6)
    assert ratio[-1] == pytest.approx(float(inputs.saturated_ratio(inputs.temperature(times[-1]))), rel=0.01)


def test_refactored_model_ends_at_saturation(case, states):
    inputs, _, times = case
    final = measurements.dissolved_ratio(states["refactored"])[-1]  # [kg/kg]

    assert final == pytest.approx(float(inputs.saturated_ratio(inputs.temperature(times[-1]))), rel=1e-3)


def test_both_models_give_the_same_predictions(case, states):
    inputs = case[0]
    for name, prediction in (
            ("dissolved ratio", measurements.dissolved_ratio),
            ("solid mass", lambda state: measurements.solid_mass(state, inputs.solute_density, inputs.shape_factor)),
            ("count signal", lambda state: measurements.count_signal(state, 1, 1.0)),
            ("Sauter mean", measurements.mean_size)):
        np.testing.assert_allclose(prediction(states["refactored"]), prediction(states["legacy"]),
                                   rtol=PARITY_TOLERANCE, err_msg=name)


def test_predictions_respond_to_the_kinetic_parameters_and_repeat_exactly(case, states):
    inputs, kinetics, times = case
    base = states["legacy"]
    middle = int(np.argmin(np.abs(times - 3000.0)))  # during the desupersaturation

    faster = batch_cooling.simulate(inputs, dict(kinetics, growth=(0.1, 0.0, 1.0)), times, "legacy")
    more_nuclei = batch_cooling.simulate(inputs, dict(kinetics, secondary_nucleation=(1e12, 0.0, 2.0, 1.0)), times, "legacy")
    again = batch_cooling.simulate(inputs, kinetics, times, "legacy")

    # Doubling the growth constant lowers the dissolved ratio there from 0.0393 to 0.0358.
    assert measurements.dissolved_ratio(faster)[middle] < 0.95 * measurements.dissolved_ratio(base)[middle]
    # The starting kinetics add 11 % to the 2.6e7 seed crystals; a hundredfold
    # nucleation constant gives 2.5e8 crystals.
    assert base.moments[-1, 0] == pytest.approx(1.11 * base.moments[0, 0], rel=0.02)
    assert more_nuclei.moments[-1, 0] > 5.0 * base.moments[-1, 0]
    # A, B, A: the first parameters give the first result again, bit for bit.
    np.testing.assert_array_equal(again.moments, base.moments)
    np.testing.assert_array_equal(again.solute_mass, base.solute_mass)


def test_every_fbrm_sample_time_gets_a_row(case):
    inputs, kinetics, _ = case
    times = batch_cooling.measurement_times(CASE, RUN, "fbrm")  # 3422 samples, 10 s apart

    state = batch_cooling.simulate(inputs, kinetics, times, "legacy")

    np.testing.assert_array_equal(state.time, times)
    assert np.all(np.isfinite(measurements.count_signal(state, 1, 1.0)))
