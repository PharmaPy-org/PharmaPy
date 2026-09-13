"""Solver-free tests for connection state-shape and interpolation helpers."""

import numpy as np
import pytest

from PharmaPy.Connections import (
    convert_str_flowsheet,
    get_input_dict,
    get_missing_field,
    get_remaining_states,
    interpolate_inputs,
)
from PharmaPy.MixedPhases import Slurry
from PharmaPy.Phases import LiquidPhase, SolidPhase


pytestmark = pytest.mark.unit


def test_convert_str_flowsheet_builds_adjacency_map_from_sequence():
    flowsheet = "Feed --> Reactor --> Filter"

    result = convert_str_flowsheet(flowsheet)

    assert result == {
        "Feed": ["Reactor"],
        "Reactor": ["Filter"],
        "Filter": [],
    }


@pytest.mark.parametrize(
    ("dim_state", "n_times", "expected"),
    [
        (1, 1, 0),
        (3, 1, np.zeros(3)),
        (1, 2, np.zeros(2)),
        (2, 3, np.zeros((3, 2))),
    ],
)
def test_get_missing_field_returns_zero_with_expected_shape(
    dim_state,
    n_times,
    expected,
):
    result = get_missing_field(dim_state, n_times)

    np.testing.assert_allclose(result, expected)


def test_get_input_dict_splits_vector_input_by_state_dimensions():
    input_data = np.array([1.0, 2.0, 300.0])
    name_dict = {"Inlet": {"mole_conc": 2, "temp": 1}}

    result = get_input_dict(input_data, name_dict)

    np.testing.assert_allclose(result["Inlet"]["mole_conc"], [1.0, 2.0])
    assert result["Inlet"]["temp"] == pytest.approx(300.0)


def test_get_input_dict_splits_time_series_columns_by_state_dimensions():
    input_data = np.array([
        [1.0, 2.0, 300.0],
        [3.0, 4.0, 310.0],
    ])
    name_dict = {"Inlet": {"mole_conc": 2, "temp": 1}}

    result = get_input_dict(input_data, name_dict)

    np.testing.assert_allclose(result["Inlet"]["mole_conc"], input_data[:, :2])
    np.testing.assert_allclose(result["Inlet"]["temp"], [300.0, 310.0])


def test_interpolate_inputs_evaluates_vector_times_without_solver_stack():
    inlet_time = np.array([0.0, 1.0, 2.0])
    inlet_values = np.array([
        [0.0, 0.0],
        [1.0, 10.0],
        [2.0, 20.0],
    ])

    result = interpolate_inputs(
        np.array([0.5, 1.5]),
        inlet_time,
        inlet_values,
    )

    np.testing.assert_allclose(result, [[0.5, 5.0], [1.5, 15.0]])


def test_get_remaining_states_uses_stream_values_and_zero_defaults(data_path):
    """Broadcast real scalar/vector stream states and fill missing values.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.
    """
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    liquid = LiquidPhase(
        thermo_path,
        temp=300.0,  # [K]
        mass=1.0,  # [kg]
        mass_frac=np.array([0.2, 0.3, 0.0, 0.5]),  # [-]
        verbose=False,
    )
    solid = SolidPhase(
        thermo_path,
        temp=300.0,  # [K]
        moments=np.array([1.0, 0.0, 0.0, 1.0e-6]),  # [m**n]
        mass_frac=np.array([1.0, 0.0, 0.0, 0.0]),  # [-]
    )
    stream = Slurry()
    stream.Phases = [liquid, solid]

    state_dimensions = {
        "Inlet": {
            "moments": 4,
            "mass_slurry": 1,
            "temp": 1,
            "missing": 2,
        },
        "Solid_1": {"mu_n": 3},
    }
    existing_inlets = {"Inlet": {"temp": 305.0}, "Solid_1": {}}

    result = get_remaining_states(
        state_dimensions,
        stream,
        existing_inlets,
        time=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(
        result["Inlet"]["moments"],
        np.tile(stream.moments, (2, 1)),  # [m**n]
    )
    np.testing.assert_allclose(
        result["Inlet"]["mass_slurry"],
        [stream.mass_slurry, stream.mass_slurry],  # [kg]
    )
    np.testing.assert_allclose(result["Inlet"]["missing"], np.zeros((2, 2)))
    np.testing.assert_allclose(
        result["Solid_1"]["mu_n"], np.zeros((2, 3))
    )
