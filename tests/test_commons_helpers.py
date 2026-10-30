"""Solver-free checks of shared ``PharmaPy.Commons`` helpers.

State packing and unpacking, finite-volume fluxes against hand
calculations, and the estimation row selection ``_requested_rows`` shared by
the reactor and crystallizer ``paramest_wrapper`` on hand-built reported
time profiles. Fast unit tests without the optional Assimulo stack.
"""

import numpy as np
import pytest

from PharmaPy.Commons import (
    _requested_rows,
    flatten_states,
    high_resolution_fvm,
    unpack_discretized,
    unpack_states,
    upwind_fvm,
)


pytestmark = pytest.mark.unit


def test_unpack_states_splits_vector_state_by_declared_dimensions():
    states = np.array([1.0, 2.0, 300.0, 4.0])

    result = unpack_states(
        states,
        num_states=[2, 1, 1],
        name_states=["mole_conc", "temp", "vol"],
    )

    np.testing.assert_allclose(result["mole_conc"], [1.0, 2.0])
    assert result["temp"] == pytest.approx(300.0)
    assert result["vol"] == pytest.approx(4.0)


def test_unpack_states_splits_time_series_and_applies_state_map():
    states = np.array([
        [1.0, 2.0, 300.0, 4.0],
        [5.0, 6.0, 310.0, 8.0],
    ])

    result = unpack_states(
        states,
        num_states=[2, 1, 1],
        name_states=["mole_conc", "temp", "vol"],
        state_map=[True, False, True],
    )

    assert list(result) == ["mole_conc", "vol"]
    np.testing.assert_allclose(result["mole_conc"], states[:, :2])
    np.testing.assert_allclose(result["vol"], [4.0, 8.0])


def test_unpack_discretized_rebuilds_finite_volume_state_profiles():
    states = np.array([
        [1.0, 10.0, 100.0, 2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0, 4.0, 40.0, 400.0],
    ])

    result = unpack_discretized(
        states,
        num_states=[1, 2],
        name_states=["temp", "mole_conc"],
        indexes={"temp": None, "mole_conc": ["A", "B"]},
    )

    np.testing.assert_allclose(result["temp"], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        result["mole_conc"]["A"],
        [[10.0, 20.0], [30.0, 40.0]],
    )
    np.testing.assert_allclose(
        result["mole_conc"]["B"],
        [[100.0, 200.0], [300.0, 400.0]],
    )


def test_flatten_states_merges_profiles_without_duplicate_boundaries():
    segments = [
        {
            "time": np.array([0.0, 1.0]),
            "states": np.array([[1.0, 10.0], [2.0, 20.0]]),
        },
        {
            "time": np.array([1.0, 2.0, 3.0]),
            "states": np.array([[2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),
        },
    ]

    result = flatten_states(segments)

    np.testing.assert_allclose(result["time"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        result["states"],
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
    )


def test_high_resolution_fvm_van_leer_flux_matches_hand_calculation():
    state = np.array([1.0, 2.0, 4.0])

    fluxes = high_resolution_fvm(state, boundary_cond=0.0)

    np.testing.assert_allclose(fluxes, [0.0, 1.5, 8.0 / 3.0, 5.0])


def test_high_resolution_fvm_supports_one_physical_cell():
    """Use a zero-gradient outlet ghost value for one physical cell."""
    cell_temperature = np.array([300.0])  # [K]

    face_temperature = high_resolution_fvm(
        cell_temperature, boundary_cond=295.0)  # [K]

    np.testing.assert_allclose(face_temperature, [295.0, 300.0])


def test_high_resolution_fvm_rejects_unknown_limiter():
    """Reject an unsupported finite-volume limiter explicitly."""
    with pytest.raises(ValueError, match="supports only the 'Van Leer'"):
        high_resolution_fvm(
            np.array([1.0, 2.0]),  # [-]
            boundary_cond=0.0,
            limiter_type="unknown",
        )


def test_high_resolution_fvm_rejects_empty_grid():
    """Reject a finite-volume call with no physical cells."""
    with pytest.raises(ValueError, match="requires at least one cell"):
        high_resolution_fvm(np.empty(0), boundary_cond=0.0)


def test_upwind_fvm_returns_neighbor_differences_from_boundary():
    state = np.array([1.0, 2.0, 4.0])

    fluxes = upwind_fvm(state, boundary_cond=0.0)

    np.testing.assert_allclose(fluxes, [1.0, 1.0, 2.0])


def test_requested_rows_maps_kth_replicate_to_kth_reported_row():
    # Hand-built solver profile [s]: the initial time once (CVode merges
    # initial replicates) and 2 s reported twice.
    reported = np.array([0.0, 1.0, 2.0, 2.0, 3.0])
    # Out-of-order requests [s] with three replicates of 2 s, one more than
    # reported, and two of the initial time.
    requested = np.array([2.0, 0.0, 2.0, 3.0, 2.0, 0.0])

    rows = _requested_rows(reported, requested, 'crystallizer')

    # 2 s: first and second replicates -> rows 2 and 3, the extra third ->
    # the last such row 3; 0 s -> the single initial row 0; 3 s -> row 4.
    np.testing.assert_array_equal(rows, [2, 0, 3, 4, 3, 0])
    assert rows.dtype.kind == 'i'


@pytest.mark.parametrize('label', ['reactor', 'crystallizer'])
def test_requested_rows_names_the_unit_for_an_absent_time(label):
    with pytest.raises(ValueError, match=(
            rf"^Requested time 1\.5 s is absent from the {label} solution, "
            r"which reports 3 times from 0\.0 s to 2\.0 s\.")):
        _requested_rows([0.0, 1.0, 2.0], [1.0, 1.5], label)


def test_requested_rows_reports_an_empty_profile():
    with pytest.raises(ValueError, match=(
            r"^Requested time 1\.0 s is absent from the crystallizer "
            r"solution, which reports no times\.")):
        _requested_rows([], [1.0], 'crystallizer')
