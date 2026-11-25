"""Backend-free validation of the provisional Clapeyron density query."""

from importlib.util import find_spec
import subprocess
import sys

import numpy as np
import pytest

from PharmaPy._clapeyron_density import single_phase_density


pytestmark = pytest.mark.unit

# IAPWS R6-95(2018), Table 7, first liquid water condition. Convert MPa to Pa.
WATER_TEMPERATURE = 300.0  # [K]
WATER_PRESSURE = 99_241.8352  # [Pa]


@pytest.mark.parametrize(
    ("model", "components", "phase", "error"),
    [
        ("unknown", ("water",), "liquid", "model must"),
        ("IAPWS95", ("ethanol",), "liquid", "IAPWS95 requires"),
        ("PR", ("ethane", "ethane"), "vapor", "PR requires"),
        ("PR", ("methane", "propane"), "vapor", "PR requires"),
        ("IAPWS95", ("water",), "stable", "phase must"),
    ],
)
def test_model_component_and_phase_choices(model, components, phase, error):
    with pytest.raises(ValueError, match=error):
        single_phase_density(
            model, components, WATER_TEMPERATURE, WATER_PRESSURE, [1.0], phase=phase,
        )


@pytest.mark.parametrize("components", [{"methane", "ethane"}, {"methane": 0, "ethane": 1}, None])
def test_density_requires_an_ordered_component_sequence(components):
    """Reject inputs that cannot establish component-to-fraction identity."""
    with pytest.raises(ValueError, match="components must be an ordered sequence"):
        single_phase_density(
            "PR", components, WATER_TEMPERATURE, WATER_PRESSURE, [0.2, 0.8],
            phase="vapor",
        )


def test_density_rejects_masked_composition():
    """A masked fraction leaves the physical composition unknown."""
    fractions = np.ma.array([0.2, 0.8], mask=[True, False])  # [-]
    with pytest.raises(ValueError, match="mole_fractions.*masked"):
        single_phase_density(
            "PR", ("methane", "ethane"), WATER_TEMPERATURE, WATER_PRESSURE,
            fractions, phase="vapor",
        )


@pytest.mark.parametrize("quantity", ["temperature", "pressure"])
@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf, [300.0], True])
def test_state_requires_finite_positive_scalars(quantity, invalid):
    """Reject nonphysical, profile-shaped and boolean states at the boundary."""
    state = {"temperature": WATER_TEMPERATURE, "pressure": WATER_PRESSURE}  # [K], [Pa]
    state[quantity] = invalid  # Same units as the input under test.
    with pytest.raises(ValueError, match=rf"{quantity} .*finite, positive real scalar"):
        single_phase_density("IAPWS95", ("water",), mole_fractions=[1.0], phase="liquid", **state)


@pytest.mark.parametrize(
    ("fractions", "error"),
    [
        (1.0, "one real value per component"),
        ([[1.0]], "one real value per component"),
        ([], "one real value per component"),
        ([1.0, 0.0], "one real value per component"),
        ([1.0 + 0.0j], "one real value per component"),
        (["1.0"], "one real value per component"),
        ([np.inf], "finite values"),
        ([np.nan], "finite values"),
        ([-0.1], "finite values"),
        ([1.1], "finite values"),
        ([0.9], "sum to one"),
    ],
)
def test_mole_fractions_require_one_normalized_vector(fractions, error):
    """Reject dimensionless inputs outside the per-component mole basis."""
    with pytest.raises(ValueError, match=error):
        single_phase_density(
            "IAPWS95", ("water",), WATER_TEMPERATURE, WATER_PRESSURE, fractions,
            phase="liquid",
        )


def test_density_module_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import PharmaPy._clapeyron_density; "
            "assert 'pyclapeyron' not in sys.modules and 'juliacall' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.skipif(find_spec("pyclapeyron") is not None, reason="requires a backend-free environment")
def test_missing_wrapper_reports_how_to_configure_backend():
    with pytest.raises(ImportError, match="requires pyclapeyron and a configured Julia"):
        single_phase_density(
            "IAPWS95", ("water",), WATER_TEMPERATURE, WATER_PRESSURE, [1.0],
            phase="liquid",
        )
