"""Species-axis assembly in ``VaporPhase.getEnthalpy``.

Regression coverage for issue #177.  When a vapor phase contains both
supercritical and subcritical species, ``total_h=False`` must return one column
per species in the phase's original order for scalar and one-element-array
temperature inputs.  The expected sensible and latent enthalpies below are
calculated directly from constant heat capacities and the Watson correlation
rather than through the implementation under test.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Phases import VaporPhase


pytestmark = pytest.mark.unit


# Synthetic pure-component data chosen only to exercise the mixed-criticality
# branch; the values do not represent a measured dataset.  The species order is
# deliberately subcritical, supercritical, subcritical at the probe temperature
# so omitting the species reordering changes the returned values.
#
# Schema units: ``mw`` [g/mol], ``t_crit`` [K], ``rho_liq`` [kg/m**3],
# ``cp_liq`` and ``cp_vapor`` constant coefficients [J/mol/K], Antoine
# coefficients ``p_vap`` [A: -, B: K, C: K], ``delta_hvap`` [J/mol], and
# ``tref_hvap`` [K].  Density and Antoine values satisfy constructor data
# requirements but are not used by these enthalpy tests.
PURE_COMPONENTS = {
    "heavy": {
        "mw": 100.0,
        "t_crit": 700.0,
        "rho_liq": 900.0,
        "cp_liq": [150.0],
        "cp_vapor": [80.0],
        "p_vap": [8.0, 1800.0, -40.0],
        "delta_hvap": 60000.0,
        "tref_hvap": 350.0,
    },
    "carrier": {
        "mw": 18.0,
        "t_crit": 650.0,
        "rho_liq": 1000.0,
        "cp_liq": [75.0],
        "cp_vapor": [35.0],
        "p_vap": [8.0, 1500.0, -40.0],
        "delta_hvap": 40000.0,
        "tref_hvap": 350.0,
    },
    "medium": {
        "mw": 46.0,
        "t_crit": 720.0,
        "rho_liq": 850.0,
        "cp_liq": [110.0],
        "cp_vapor": [60.0],
        "p_vap": [8.0, 1650.0, -40.0],
        "delta_hvap": 50000.0,
        "tref_hvap": 350.0,
    },
}

SPECIES_NAMES = tuple(PURE_COMPONENTS)
MOLAR_MASS = np.array([100.0, 18.0, 46.0])  # [g/mol]
CRITICAL_TEMPERATURE = np.array([700.0, 650.0, 720.0])  # [K]
CP_LIQUID = np.array([150.0, 75.0, 110.0])  # [J/mol/K]
CP_VAPOR = np.array([80.0, 35.0, 60.0])  # [J/mol/K]
LATENT_HEAT_REFERENCE = np.array([60000.0, 40000.0, 50000.0])  # [J/mol]
LATENT_HEAT_REFERENCE_TEMPERATURE = np.full(3, 350.0)  # [K]

PROBE_TEMPERATURE = 660.0  # [K]
ENTHALPY_REFERENCE_TEMPERATURE = 298.15  # [K]
# Correlation exponent documented by ``VaporPhase.getHeatVaporization``.
WATSON_EXPONENT = 0.38  # [-]
GRAMS_PER_KILOGRAM = 1000.0  # [g/kg], exact unit conversion


def _make_vapor_phase(tmp_path: Path, num_species: int) -> VaporPhase:
    """Build an equimolar synthetic vapor phase.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory in which to write the component table.
    num_species : int
        Number of leading entries from ``SPECIES_NAMES`` to include [-].

    Returns
    -------
    VaporPhase
        Vapor phase at ``PROBE_TEMPERATURE`` [K] with equimolar composition
        and one total mole [mol].
    """
    component_data = {
        name: PURE_COMPONENTS[name] for name in SPECIES_NAMES[:num_species]
    }
    component_path = tmp_path / f"thermo_{num_species}_species.json"
    component_path.write_text(json.dumps(component_data))

    mole_fraction = np.full(num_species, 1.0 / num_species)  # [-]
    return VaporPhase(
        str(component_path),
        temp=PROBE_TEMPERATURE,
        moles=1.0,
        mole_frac=mole_fraction,
        check_input=False,
    )


def _expected_molar_enthalpy(num_species: int) -> np.ndarray:
    """Calculate per-species vapor enthalpy independently.

    Parameters
    ----------
    num_species : int
        Number of leading fixture species to evaluate [-].

    Returns
    -------
    ndarray
        Per-species enthalpies in phase order [J/mol], with shape
        ``(num_species,)``.

    Notes
    -----
    Supercritical species use vapor heat capacity and no latent heat.
    Subcritical species use liquid heat capacity plus latent heat extrapolated
    from its tabulated reference with the Watson correlation.
    """
    expected_enthalpy = np.empty(num_species)  # [J/mol]
    temperature_change = (
        PROBE_TEMPERATURE - ENTHALPY_REFERENCE_TEMPERATURE
    )  # [K]

    for species_index in range(num_species):
        if PROBE_TEMPERATURE > CRITICAL_TEMPERATURE[species_index]:
            expected_enthalpy[species_index] = (
                CP_VAPOR[species_index] * temperature_change
            )
        else:
            watson_ratio = (
                (CRITICAL_TEMPERATURE[species_index] - PROBE_TEMPERATURE)
                / (
                    CRITICAL_TEMPERATURE[species_index]
                    - LATENT_HEAT_REFERENCE_TEMPERATURE[species_index]
                )
            )  # [-]
            latent_heat = (
                LATENT_HEAT_REFERENCE[species_index]
                * watson_ratio**WATSON_EXPONENT
            )  # [J/mol]
            expected_enthalpy[species_index] = (
                CP_LIQUID[species_index] * temperature_change + latent_heat
            )

    return expected_enthalpy


@pytest.mark.parametrize(
    "temperature",
    [PROBE_TEMPERATURE, np.array([PROBE_TEMPERATURE])],
    ids=["scalar", "one_element_array"],
)
@pytest.mark.parametrize("basis", ["mole", "mass"])
def test_equal_subsets_return_one_ordered_species_row(
        tmp_path, basis, temperature):
    """An equal split cannot silently turn species into temperature rows.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the synthetic component table.
    basis : {'mole', 'mass'}
        Physical basis of the requested and expected enthalpy.
    temperature : float or ndarray
        Scalar or one-element-array evaluation temperature [K].
    """
    num_species = 2
    phase = _make_vapor_phase(tmp_path, num_species)
    expected_mole = _expected_molar_enthalpy(num_species)  # [J/mol]
    expected = (
        expected_mole
        if basis == "mole"
        else expected_mole * GRAMS_PER_KILOGRAM / MOLAR_MASS[:num_species]
    )  # [J/mol] or [J/kg], selected by basis

    observed = phase.getEnthalpy(
        temp=temperature, total_h=False, basis=basis
    )  # [J/mol] or [J/kg], selected by basis

    assert observed.shape == (1, num_species)
    np.testing.assert_allclose(observed[0], expected, rtol=1e-12)


@pytest.mark.parametrize(
    "temperature",
    [PROBE_TEMPERATURE, np.array([PROBE_TEMPERATURE])],
    ids=["scalar", "one_element_array"],
)
@pytest.mark.parametrize("basis", ["mole", "mass"])
def test_unequal_subsets_keep_every_species_column(
        tmp_path, basis, temperature):
    """One supercritical and two subcritical species remain aligned.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the synthetic component table.
    basis : {'mole', 'mass'}
        Physical basis of the requested and expected enthalpy.
    temperature : float or ndarray
        Scalar or one-element-array evaluation temperature [K].
    """
    num_species = 3
    phase = _make_vapor_phase(tmp_path, num_species)
    expected_mole = _expected_molar_enthalpy(num_species)  # [J/mol]
    expected = (
        expected_mole
        if basis == "mole"
        else expected_mole * GRAMS_PER_KILOGRAM / MOLAR_MASS
    )  # [J/mol] or [J/kg], selected by basis

    observed = phase.getEnthalpy(
        temp=temperature, total_h=False, basis=basis
    )  # [J/mol] or [J/kg], selected by basis

    assert observed.shape == (1, num_species)
    np.testing.assert_allclose(observed[0], expected, rtol=1e-12)


@pytest.mark.parametrize(
    "temperature",
    [PROBE_TEMPERATURE, np.array([PROBE_TEMPERATURE])],
    ids=["scalar", "one_element_array"],
)
@pytest.mark.parametrize("basis", ["mole", "mass"])
def test_total_enthalpy_sibling_remains_fraction_weighted(
        tmp_path, basis, temperature):
    """The existing mixture branch keeps its fraction-weighted value.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the synthetic component table.
    basis : {'mole', 'mass'}
        Physical basis of the requested and expected enthalpy.
    temperature : float or ndarray
        Scalar or one-element-array evaluation temperature [K].
    """
    num_species = 3
    phase = _make_vapor_phase(tmp_path, num_species)
    expected_mole = _expected_molar_enthalpy(num_species)  # [J/mol]

    if basis == "mole":
        expected_total = np.dot(phase.mole_frac, expected_mole)  # [J/mol]
    else:
        expected_mass = (
            expected_mole * GRAMS_PER_KILOGRAM / MOLAR_MASS
        )  # [J/kg]
        expected_total = np.dot(phase.mass_frac, expected_mass)  # [J/kg]

    observed_total = phase.getEnthalpy(
        temp=temperature, total_h=True, basis=basis
    )  # [J/mol] or [J/kg], selected by basis

    assert observed_total == pytest.approx(expected_total, rel=1e-12)
