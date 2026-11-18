"""Measurement predictions from a slurry state, and the two model-layout adapters.

One physical state is written by hand in both layouts, with unequal species
positions, so that an adapter that confuses a basis, a unit or a column fails.
"""
import numpy as np
import pytest

from PharmaPy import SlurryMeasurements as measurements
from PharmaPy.Errors import PharmaPyValueError

pytestmark = pytest.mark.unit

TIME = [10.0, 20.0]  # [s]
# Total moments, order k in [um**k]; in SI they are (2e6, 400 m, 0.1 m2, 3e-5 m3)
# and (3e6, 900 m, 0.4 m2, 2e-4 m3).
MOMENTS_UM = np.array([[2e6, 4e8, 1e11, 3e13], [3e6, 9e8, 4e11, 2e14]])
# Species order: solvent, a bystander, solute.
CONCENTRATION = np.array([[900.0, 5.0, 40.0], [910.0, 6.0, 20.0]])  # [kg/m**3]
LIQUID_VOLUME = np.array([2e-3, 1.9e-3])  # [m**3]
SHAPE_FACTOR = 0.5  # [-]; crystals take 1.5e-5 and 1e-4 m3
SLURRY_VOLUME = np.array([2.015e-3, 2.0e-3])  # [m**3]
SOLUTE_DENSITY = 1200.0  # [kg/m**3]


@pytest.fixture()
def state():
    return measurements.state_from_total_moments(
        TIME, np.column_stack((MOMENTS_UM, CONCENTRATION, LIQUID_VOLUME)), solute_index=2, solvent_index=0,
        shape_factor=SHAPE_FACTOR)


def test_total_moment_layout_gives_si_moments_masses_and_slurry_volume(state):
    np.testing.assert_allclose(state.moments, [[2e6, 400.0, 0.1, 3e-5], [3e6, 900.0, 0.4, 2e-4]])
    np.testing.assert_allclose(state.solute_mass, [0.08, 0.038])
    np.testing.assert_allclose(state.solvent_mass, [1.8, 1.729])
    np.testing.assert_allclose(state.slurry_volume, SLURRY_VOLUME)
    np.testing.assert_array_equal(state.time, TIME)


def test_specific_moment_layout_gives_the_same_state(state):
    other = measurements.state_from_specific_moments(
        TIME, MOMENTS_UM / SLURRY_VOLUME[:, np.newaxis], CONCENTRATION * LIQUID_VOLUME[:, np.newaxis], SLURRY_VOLUME,
        solute_index=2, solvent_index=0)

    for name in ("time", "moments", "solute_mass", "solvent_mass", "slurry_volume"):
        np.testing.assert_allclose(getattr(other, name), getattr(state, name), rtol=1e-12, err_msg=name)


def test_predictions_follow_from_the_state(state):
    np.testing.assert_allclose(measurements.dissolved_ratio(state), [0.08 / 1.8, 0.038 / 1.729])
    np.testing.assert_allclose(measurements.solid_mass(state, SOLUTE_DENSITY, SHAPE_FACTOR), [0.018, 0.12])
    np.testing.assert_allclose(measurements.total_solute(state, SOLUTE_DENSITY, SHAPE_FACTOR), [0.098, 0.158])
    # Order 1 with a constant of 2: 2 * 400 / 2.015e-3 and 2 * 900 / 2.0e-3.
    np.testing.assert_allclose(measurements.count_signal(state, 1, 2.0), [397022.3325, 900000.0])
    np.testing.assert_allclose(measurements.count_signal(state, 0, 0.5), [0.5 * 2e6 / 2.015e-3, 0.5 * 3e6 / 2.0e-3])
    np.testing.assert_allclose(measurements.mean_size(state), [3e-4, 5e-4])  # Sauter mean [m]
    np.testing.assert_allclose(measurements.mean_size(state, upper=1, lower=0), [2e-4, 3e-4])


def test_mean_size_is_undefined_without_crystals():
    empty = measurements.state_from_specific_moments([0.0], np.zeros((1, 4)), [[1.0, 0.1]], [1e-3], solute_index=1,
                                                     solvent_index=0)

    assert np.isnan(measurements.mean_size(empty)[0])
    assert measurements.count_signal(empty, 2, 3.0)[0] == 0.0


def test_inputs_that_do_not_fit_are_refused(state):
    states = np.column_stack((MOMENTS_UM, CONCENTRATION, LIQUID_VOLUME))
    with pytest.raises(PharmaPyValueError, match="At least four moments"):
        measurements.state_from_total_moments(TIME, states, 2, 0, SHAPE_FACTOR, moment_count=3)
    with pytest.raises(PharmaPyValueError, match="do not hold 4 moments"):
        measurements.state_from_total_moments([10.0], states, 2, 0, SHAPE_FACTOR)
    with pytest.raises(PharmaPyValueError, match="do not match 2 times"):
        measurements.state_from_specific_moments(TIME, MOMENTS_UM, CONCENTRATION, [1e-3], 2, 0)
    with pytest.raises(PharmaPyValueError, match="order 4 is not among them"):
        measurements.count_signal(state, 4, 1.0)
    with pytest.raises(PharmaPyValueError, match="not consecutive"):
        measurements.mean_size(state, upper=3, lower=1)
    # Three species: position 3 does not exist, and one species cannot be both.
    with pytest.raises(PharmaPyValueError, match="Solute index 3 and solvent index 0"):
        measurements.state_from_total_moments(TIME, states, 3, 0, SHAPE_FACTOR)
    with pytest.raises(PharmaPyValueError, match="Solute index 1 and solvent index 1"):
        measurements.state_from_specific_moments(TIME, MOMENTS_UM, CONCENTRATION, SLURRY_VOLUME, 1, 1)
    with pytest.raises(PharmaPyValueError, match="shape_factor must be finite and positive; got 0.0"):
        measurements.state_from_total_moments(TIME, states, 2, 0, 0.0)
    with pytest.raises(PharmaPyValueError, match="solute_density must be finite and positive; got -1200.0"):
        measurements.solid_mass(state, -SOLUTE_DENSITY, SHAPE_FACTOR)
    with pytest.raises(PharmaPyValueError, match="shape_factor must be finite and positive; got nan"):
        measurements.total_solute(state, SOLUTE_DENSITY, float("nan"))
