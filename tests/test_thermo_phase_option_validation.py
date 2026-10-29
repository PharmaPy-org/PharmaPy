"""Validate the ``phase`` option and inputs of the shared property kernels.

Issue #421: the density kernels ``ThermoPhysicalManager.getDensityPure`` and
``getDensityMix`` accept only ``'liquid'`` and ``'solid'``; any other value,
including ``'vapor'``, ``None`` and a miscased ``'Liquid'``, raises
``ValueError`` in the format of the viscosity kernels before any computation,
where earlier releases raised ``UnboundLocalError`` (``getDensityPure``) or an
unrelated ``AttributeError`` (``getDensityMix`` without a ``temp`` attribute).
The heat-capacity kernels (``getCpPure``, ``getCpMix``, ``getEnthalpy``)
already validate their three phases (PR #404); their miscased and ``None``
cases are covered here alongside. ``getEnthalpy(total_h=True)`` without any
composition raises a ``ValueError`` naming the missing input, and the
temperature-independent density methods no longer need a ``temp`` attribute.

The fixture is a synthetic three-species database (rounded test values, not
measured properties) with unequal molar masses, liquid and solid densities and
heat-capacity polynomials, so a swapped species, phase or basis changes every
result. Expected values are derived independently from the fixture data:
densities as the mass or moles of a 1 kg sample over its additive volume, and
enthalpies as closed-form integrals of the heat-capacity polynomials. A bare
``ThermoPhysicalManager`` (no ``temp`` attribute) is used; no solver backend is
needed.
"""

import json

import numpy as np
import pytest

from PharmaPy.ThermoModule import ThermoPhysicalManager


pytestmark = pytest.mark.unit

# Synthetic species data in database order. Units: mw [g/mol], rho_liq and
# rho_solid [kg/m**3], cp_liq and cp_solid ascending polynomial coefficients
# [J/mol/K/K**k].
MOLAR_MASS = np.array([46.0, 180.0, 58.5])  # [g/mol]
RHO_LIQUID = np.array([789.0, 1250.0, 1500.0])  # [kg/m**3]
RHO_SOLID = np.array([950.0, 1400.0, 2160.0])  # [kg/m**3]
CP_LIQUID = ([60.0, 0.15], [300.0, 0.0], [80.0, 0.05])  # [J/mol/K], [J/mol/K**2]
CP_SOLID = (40.0, 250.0, 50.0)  # [J/mol/K], constant
DATABASE = {
    name: {
        "mw": MOLAR_MASS[ind],
        "rho_liq": RHO_LIQUID[ind],
        "rho_solid": RHO_SOLID[ind],
        "cp_liq": list(CP_LIQUID[ind]),
        "cp_solid": [CP_SOLID[ind]],
    }
    for ind, name in enumerate(("solvent", "solute", "salt"))
}

MASS_FRACTIONS = np.array([0.6, 0.3, 0.1])  # [-], unequal, sum to one
MOLE_FRACTIONS = np.array([0.5, 0.2, 0.3])  # [-], unequal, sum to one
TEMPERATURE = 330.0  # [K]
TEMP_REF = 298.15  # [K], documented getEnthalpy default reference
# [K], a second, arbitrary temperature: stored densities must not depend on it
OTHER_TEMPERATURE = 450.0
SAMPLE_MASS = 1.0  # [kg], any positive amount; densities are intensive
GRAMS_PER_KILOGRAM = 1000.0  # [g/kg], exact unit conversion
MOLES_PER_KILOMOLE = 1000.0  # [mol/kmol], exact unit conversion

# Expected values are closed-form expressions of the same data, so they agree
# with PharmaPy to floating-point roundoff.
RTOL = 1e-12  # [-]

# A miscased accepted value, a synonym, None and, for the densities only,
# 'vapor', which has no stored pure density.
INVALID_CP_PHASES = ("Liquid", "gas", None)
INVALID_DENSITY_PHASES = ("Liquid", "gas", None, "vapor")


@pytest.fixture
def manager(tmp_path):
    """Build a bare property manager on the synthetic database.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test temporary directory.

    Returns
    -------
    ThermoPhysicalManager
        Manager without a ``temp`` attribute.
    """
    path = tmp_path / "phase_option.json"
    path.write_text(json.dumps(DATABASE))
    built = ThermoPhysicalManager(str(path))
    assert not hasattr(built, "temp")
    return built


@pytest.mark.parametrize("phase", INVALID_DENSITY_PHASES)
@pytest.mark.parametrize("call", [
    lambda manager, phase: manager.getDensityPure(phase=phase),
    lambda manager, phase: manager.getDensityPure(phase, TEMPERATURE),
    lambda manager, phase: manager.getDensityMix(phase=phase),
    lambda manager, phase: manager.getDensityMix(phase=phase, basis="mole"),
], ids=["pure", "pure_with_temp", "mix_mass", "mix_mole"])
def test_density_kernels_reject_unknown_phase(manager, call, phase):
    """Reject the phase before any computation, with the viscosity format.

    ``getDensityMix`` is called without fractions: a computation would fail
    converting the missing composition, so the exact ``ValueError`` shows
    that the phase is checked first.
    """
    with pytest.raises(ValueError) as error:
        call(manager, phase)

    assert str(error.value) == (
        f"phase must be one of ('liquid', 'solid'), got {phase!r}")


@pytest.mark.parametrize("phase", INVALID_CP_PHASES)
@pytest.mark.parametrize("call", [
    lambda manager, phase: manager.getCpPure(TEMPERATURE, phase=phase),
    lambda manager, phase: manager.getCpMix(TEMPERATURE, phase=phase,
                                            mass_frac=MASS_FRACTIONS),
    lambda manager, phase: manager.getEnthalpy(TEMPERATURE, phase=phase,
                                               mass_frac=MASS_FRACTIONS),
    lambda manager, phase: manager.getEnthalpy(TEMPERATURE, phase=phase,
                                               total_h=False),
    lambda manager, phase: manager.getEnthalpy(TEMPERATURE, phase=phase),
], ids=["getCpPure", "getCpMix", "getEnthalpy_total", "getEnthalpy_species",
        "getEnthalpy_total_without_fractions"])
def test_heat_capacity_kernels_reject_unknown_phase(manager, call, phase):
    """Reject miscased, synonym and None phases of the Cp kernels.

    The last case also omits the fractions: the phase error is reported
    first.
    """
    with pytest.raises(ValueError) as error:
        call(manager, phase)

    assert str(error.value) == (
        f"phase must be one of ('liquid', 'solid', 'vapor'), got {phase!r}")


@pytest.mark.parametrize("phase, rho_pure", [
    ("liquid", RHO_LIQUID), ("solid", RHO_SOLID)])
@pytest.mark.parametrize("temp", [None, TEMPERATURE, OTHER_TEMPERATURE],
                         ids=["no_temp", "temp", "other_temp"])
def test_valid_densities_are_unchanged_and_temperature_independent(
        manager, phase, rho_pure, temp):
    """Accepted phases return the stored densities without a temp attribute.

    The mixture densities are the mass and moles of a 1 kg sample over the
    sum of the pure-species volumes; any temperature gives the same value.
    """
    species_mass = SAMPLE_MASS * MASS_FRACTIONS  # [kg]
    sample_volume = np.sum(species_mass / rho_pure)  # [m**3]
    species_moles = species_mass * GRAMS_PER_KILOGRAM / MOLAR_MASS  # [mol]
    expected_mass = SAMPLE_MASS / sample_volume  # [kg/m**3]
    expected_mole = (np.sum(species_moles) / MOLES_PER_KILOMOLE
                     / sample_volume)  # [kmol/m**3]

    rho_mass, rho_mole = manager.getDensityPure(phase, temp)  # [kg/m**3], [kmol/m**3]
    mix_mass = manager.getDensityMix(mass_frac=MASS_FRACTIONS, phase=phase,
                                     temp=temp)  # [kg/m**3]
    mix_mole = manager.getDensityMix(mass_frac=MASS_FRACTIONS, phase=phase,
                                     temp=temp, basis="mole")  # [kmol/m**3]

    np.testing.assert_array_equal(rho_mass, rho_pure)
    np.testing.assert_allclose(
        rho_mole, rho_pure * GRAMS_PER_KILOGRAM / MOLAR_MASS
        / MOLES_PER_KILOMOLE, rtol=RTOL)
    assert mix_mass == pytest.approx(expected_mass, rel=RTOL)
    assert mix_mole == pytest.approx(expected_mole, rel=RTOL)


def test_mixture_enthalpy_without_fractions_names_the_missing_input(manager):
    """``total_h=True`` needs a composition; say so instead of crashing."""
    with pytest.raises(ValueError) as error:
        manager.getEnthalpy(TEMPERATURE, total_h=True)

    assert str(error.value) == (
        "getEnthalpy(total_h=True) weights the species enthalpies by "
        "composition and needs mass_frac or mole_frac [-]; neither was given "
        "(pass total_h=False for species enthalpies)")


def _polynomial_enthalpy(coefficients, temp, temp_ref):
    """Integrate an ascending heat-capacity polynomial in closed form.

    Parameters
    ----------
    coefficients : sequence of float
        Ascending coefficients ``c_k`` [J/mol/K/K**k].
    temp, temp_ref : float
        Upper and lower integration limits [K].

    Returns
    -------
    float
        ``sum_k c_k / (k + 1) * (temp**(k + 1) - temp_ref**(k + 1))``
        [J/mol].
    """
    return sum(coef / (power + 1) * (temp**(power + 1) - temp_ref**(power + 1))
               for power, coef in enumerate(coefficients))


@pytest.mark.parametrize("phase, coefficients", [
    ("liquid", CP_LIQUID), ("solid", [[cp] for cp in CP_SOLID])],
    ids=["liquid_polynomial", "solid_constant"])
def test_valid_enthalpies_are_unchanged_from_either_composition(
        manager, phase, coefficients):
    """Mixture enthalpies from the converted fraction are unchanged.

    Mass-basis enthalpy from mole fractions and molar enthalpy from mass
    fractions exercise the fraction conversions inside ``getEnthalpy``;
    the solid case is the constant ``cp_solid`` integral times
    ``1000 / mw``.
    """
    molar_enthalpy = np.array([
        _polynomial_enthalpy(coefs, TEMPERATURE, TEMP_REF)
        for coefs in coefficients])  # [J/mol]
    specific_enthalpy = molar_enthalpy * GRAMS_PER_KILOGRAM / MOLAR_MASS  # [J/kg]
    # Mass fractions of the mole-fraction mixture, from 1 mol of it.
    mass_of_mole_sample = MOLE_FRACTIONS * MOLAR_MASS  # [g]
    mass_from_mole = mass_of_mole_sample / mass_of_mole_sample.sum()  # [-]
    # Mole fractions of the mass-fraction mixture, from 1 g of it.
    moles_of_mass_sample = MASS_FRACTIONS / MOLAR_MASS  # [mol]
    mole_from_mass = moles_of_mass_sample / moles_of_mass_sample.sum()  # [-]

    h_mass = manager.getEnthalpy(TEMPERATURE, mole_frac=MOLE_FRACTIONS,
                                 phase=phase)  # [J/kg]
    h_mole = manager.getEnthalpy(TEMPERATURE, mass_frac=MASS_FRACTIONS,
                                 phase=phase, basis="mole")  # [J/mol]
    h_species = manager.getEnthalpy(TEMPERATURE, phase=phase,
                                    total_h=False)  # [J/kg]

    assert h_mass == pytest.approx(specific_enthalpy @ mass_from_mole, rel=RTOL)
    assert h_mole == pytest.approx(molar_enthalpy @ mole_from_mass, rel=RTOL)
    assert h_species.shape == (1, len(MOLAR_MASS))
    np.testing.assert_allclose(h_species[0], specific_enthalpy, rtol=RTOL)
