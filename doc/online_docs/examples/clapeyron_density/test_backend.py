"""Manual parity and property-reference checks using the real Julia backend."""

from importlib.metadata import version
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from PharmaPy._clapeyron_density import Density, single_phase_density


pytestmark = pytest.mark.integration

if not (os.environ.get("PYTHON_JULIACALL_EXE")
        and os.environ.get("PYTHON_JULIACALL_PROJECT")):
    pytest.skip(
        "configure the isolated Julia environment from README.md to run the smoke checks",
        allow_module_level=True,
    )


# IAPWS water and steam FAQ, VSMOW molar mass https://iapws.org/faqs/faq1
WATER_MOLAR_MASS = 0.018015268  # [kg/mol], converted from g/mol
# IAPWS R6-95(2018), Tables 7 and 8.
WATER_PRESSURE = 99_241.8352  # [Pa], Table 7 first row converted from MPa
WATER_TEMPERATURE = 300.0  # [K], Table 7 first row
SATURATION_PRESSURE = 698.451167  # [Pa], Table 8 first column converted from MPa
SATURATION_TEMPERATURE = 275.0  # [K], Table 8 first column
PURE_WATER = [1.0]  # [-], mole basis

# Dilute vapor examples near ambient temperature, with a methane-lean mixture.
BINARY_PRESSURE = 100_000.0  # [Pa], one bar
BINARY_TEMPERATURE = 300.0  # [K], near ambient
BINARY_COMPONENTS = ("methane", "ethane")
BINARY_FRACTIONS = np.array([0.2, 0.8])  # [-], mole basis
# NIST Chemistry WebBook SRD 69, CAS 74-82-8 and 74-84-0, converted from g/mol.
METHANE_MOLAR_MASS = 0.0160425  # [kg/mol]
ETHANE_MOLAR_MASS = 0.0300690  # [kg/mol]
# Half the last digit of the pinned Clapeyron source masses (0.01 g/mol).
BINARY_MOLAR_MASS_ATOL = 0.000005  # [kg/mol]

# Allow 64 Float64 epsilons to compare independent calls of the same EOS,
# including order-dependent arithmetic and the iterative volume solve.
PARITY_RTOL = 64 * np.finfo(float).eps  # [-]


@pytest.fixture(scope="module")
def julia_densities():
    """Run the reference in a separate process with regular Julia arrays."""
    assert version("pyclapeyron") == "0.1.1"
    assert version("juliacall") == "0.9.37"
    output = subprocess.check_output(
        [
            os.environ["PYTHON_JULIACALL_EXE"],
            "--startup-file=no",
            f"--project={os.environ['PYTHON_JULIACALL_PROJECT']}",
            str(Path(__file__).with_name("smoke.jl")),
        ],
        text=True,
    )
    lines = output.splitlines()
    assert lines[0] == "versions 1.10.11 0.6.21 0.9.37"
    densities = {}  # [kg/m³], [mol/m³] per state
    for line in lines[1:]:
        label, mass, molar = line.split()  # [kg/m³], [mol/m³] for mass, molar
        densities[label] = Density(float(mass), float(molar))
    return densities


def test_iapws_single_phase_water(julia_densities):
    density = single_phase_density(  # [kg/m³], [mol/m³]
        "IAPWS95", ("water",), WATER_TEMPERATURE, WATER_PRESSURE, PURE_WATER,
        phase="liquid",
    )
    published_density = 996.5560  # [kg/m³], IAPWS R6-95(2018), Table 7 first row
    table_rounding = 0.00005  # [kg/m³], half the last printed digit
    assert density.mass_density == pytest.approx(published_density, rel=0, abs=table_rounding)
    expected_molar_density = published_density / WATER_MOLAR_MASS  # [mol/m³]
    assert density.molar_density == pytest.approx(
        expected_molar_density, rel=0, abs=table_rounding / WATER_MOLAR_MASS,
    )
    assert type(density.mass_density) is type(density.molar_density) is float
    np.testing.assert_allclose(density, julia_densities["water"], rtol=PARITY_RTOL, atol=0)


@pytest.mark.parametrize(
    ("phase", "published_mass_density", "table_rounding"),
    [
        ("liquid", 999.887406, 0.0000005),
        ("vapor", 0.00550664919, 0.000000000005),
    ],
)
def test_explicit_water_roots(phase, published_mass_density, table_rounding, julia_densities):
    """Check each homogeneous root using Table 8 mass densities and rounding [kg/m³]."""
    density = single_phase_density(  # [kg/m³], [mol/m³]
        "IAPWS95", ("water",), SATURATION_TEMPERATURE, SATURATION_PRESSURE,
        PURE_WATER, phase=phase,
    )
    assert density.mass_density == pytest.approx(
        published_mass_density, rel=0, abs=table_rounding,
    )
    np.testing.assert_allclose(density, julia_densities[phase], rtol=PARITY_RTOL, atol=0)


def test_unavailable_vapor_root_at_liquid_reference_state():
    with pytest.raises(RuntimeError, match="IAPWS95 returned an invalid density for the vapor root"):
        single_phase_density(
            "IAPWS95", ("water",), WATER_TEMPERATURE, WATER_PRESSURE, PURE_WATER,
            phase="vapor",
        )


def test_peng_robinson_binary_preserves_component_order_and_basis(julia_densities):
    state = ("PR", BINARY_COMPONENTS, BINARY_TEMPERATURE, BINARY_PRESSURE)
    density = single_phase_density(*state, BINARY_FRACTIONS, phase="vapor")  # [kg/m³], [mol/m³]
    np.testing.assert_allclose(density, julia_densities["binary"], rtol=PARITY_RTOL, atol=0)

    reversed_components = BINARY_COMPONENTS[::-1]
    same_mixture = single_phase_density(  # [kg/m³], [mol/m³]
        "PR", reversed_components, BINARY_TEMPERATURE, BINARY_PRESSURE,
        BINARY_FRACTIONS[::-1], phase="vapor",
    )
    np.testing.assert_allclose(same_mixture, density, rtol=PARITY_RTOL, atol=0)

    reversed_fractions_only = single_phase_density(  # [kg/m³], [mol/m³]
        *state, BINARY_FRACTIONS[::-1], phase="vapor",
    )
    molar_masses = np.array([METHANE_MOLAR_MASS, ETHANE_MOLAR_MASS])  # [kg/mol]
    for result, fractions in ((density, BINARY_FRACTIONS), (reversed_fractions_only, BINARY_FRACTIONS[::-1])):
        mixture_molar_mass = fractions @ molar_masses  # [kg/mol]
        measured_molar_mass = result.mass_density / result.molar_density  # [kg/mol]
        assert measured_molar_mass == pytest.approx(mixture_molar_mass, rel=0, abs=BINARY_MOLAR_MASS_ATOL)
