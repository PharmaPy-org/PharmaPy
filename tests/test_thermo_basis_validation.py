"""Test the ``basis`` option of the shared enthalpy and density kernels.

Issue #415: ``ThermoPhysicalManager.getEnthalpy`` and
``ThermoPhysicalManager.getDensityMix`` must accept only ``'mass'`` or
``'mole'``. Before the fix, an invalid basis crashed mixture enthalpies with
``UnboundLocalError`` and silently returned molar species enthalpies and molar
densities. The public ``LiquidPhase`` and ``SolidPhase`` providers are
exercised, together with the kernel itself and ``VaporPhase``, whose
pre-existing check must report the same message.

The shipped five-species database
``tests/Flowsheet/data/compound_database.json`` supplies synthetic repository
values, not measured properties of real compounds. Its species have distinct
molecular weights [g/mol] and pure densities [kg/m**3], species B has its own
liquid heat-capacity polynomial, and every species has the constant solid heat
capacity 1600 J/mol/K. The mass fractions below are unequal, so a swapped
species or a basis mix-up changes the result. Expected values are derived from
the JSON data: liquid enthalpies by SciPy quadrature of the heat-capacity
polynomial, solid enthalpies by the closed-form constant-heat-capacity
integral, and mixture densities from ideal (additive) volumes. Real phase
objects are used; no solver backend is needed.
"""

import json
import re

import numpy as np
import pytest
from scipy.integrate import quad

from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.ThermoModule import ThermoPhysicalManager


pytestmark = pytest.mark.unit

FIXTURE_NAME = 'compound_database.json'

# Established message of the sibling validators (getCpMix,
# LiquidPhase.getProps and the VaporPhase methods); matched in full.
BASIS_MESSAGE = "^" + re.escape("basis must be 'mass' or 'mole'") + "$"

# A capitalization variant of 'mass' and two plausible spellings of 'mole'.
INVALID_BASES = ('Mass', 'molar', 'mol')

GRAMS_PER_KILOGRAM = 1000.0  # [g/kg], exact unit conversion

TEMPERATURE = 320.0  # [K], above the 298.15 K default reference
DEFAULT_TEMP_REF = 298.15  # [K], documented getEnthalpy default reference

# [-], unequal mass fractions of species A, B, C, D and solvent (database
# order); they sum to one.
MASS_FRACTIONS = np.array([0.05, 0.10, 0.15, 0.30, 0.40])  # [-]
PHASE_MASS = 1.0  # [kg], any positive inventory; properties are intensive

# Rounding allowance for comparisons with the independent derivations [-].
# Polynomial quadrature, the closed-form solid integral, and the fraction
# conversions are exact to floating-point roundoff.
RTOL = 1e-10  # [-]


@pytest.fixture(scope='module')
def thermo_path(data_path):
    """Return the five-species property database path.

    Parameters
    ----------
    data_path : dict
        Shared test-data directories.

    Returns
    -------
    str
        Path of the JSON property database.
    """
    return str(data_path['flowsheet'] / FIXTURE_NAME)


@pytest.fixture(scope='module')
def thermo_data(thermo_path):
    """Load the raw property data used for independent expectations.

    Parameters
    ----------
    thermo_path : str
        Path of the JSON property database.

    Returns
    -------
    dict
        Raw JSON property entries keyed by species name, in file order.
    """
    with open(thermo_path) as stream:
        data = json.load(stream)

    # Fixture self-checks: one fraction per species, and constant solid heat
    # capacity for the closed-form integral below.
    assert len(data) == len(MASS_FRACTIONS)
    for entry in data.values():
        assert len(entry['cp_solid']) == 1

    return data


def _mw_kg(thermo_data):
    """Return species molecular weights [kg/mol] in database order.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.

    Returns
    -------
    numpy.ndarray
        Molecular weights [kg/mol], shape (num_species,).
    """
    return np.array([entry['mw'] for entry in thermo_data.values()]
                    ) / GRAMS_PER_KILOGRAM  # [kg/mol]


def _mole_fractions(thermo_data):
    """Convert ``MASS_FRACTIONS`` to mole fractions.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.

    Returns
    -------
    numpy.ndarray
        Mole fractions [-], shape (num_species,): ``(w_i / mw_i) / sum_j
        (w_j / mw_j)``.
    """
    moles_per_mass = MASS_FRACTIONS / _mw_kg(thermo_data)  # [mol/kg]
    return moles_per_mass / moles_per_mass.sum()  # [-]


def _mixture_molar_mass(thermo_data):
    """Return the mixture molar mass of ``MASS_FRACTIONS``.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.

    Returns
    -------
    float
        Molar mass [kg/mol], ``1 / sum_i (w_i / mw_i)``.
    """
    return 1 / np.sum(MASS_FRACTIONS / _mw_kg(thermo_data))  # [kg/mol]


def _liquid_molar_enthalpies(thermo_data):
    """Integrate each liquid heat-capacity polynomial by quadrature.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.

    Returns
    -------
    numpy.ndarray
        Species sensible enthalpies [J/mol] from ``DEFAULT_TEMP_REF`` to
        ``TEMPERATURE``, shape (num_species,), from ascending coefficients
        ``cp = sum_k c_k T**k`` [J/mol/K].
    """
    enthalpies = []  # [J/mol]
    for entry in thermo_data.values():
        coefficients = np.asarray(
            entry['cp_liq'])  # [J/mol/K**(k+1)], ascending k

        def cp_molar(temperature, coefficients=coefficients):
            """Return liquid cp [J/mol/K] at ``temperature`` [K]."""
            return np.polynomial.polynomial.polyval(temperature, coefficients)

        molar_enthalpy, _ = quad(cp_molar, DEFAULT_TEMP_REF,
                                 TEMPERATURE)  # [J/mol]
        enthalpies.append(molar_enthalpy)
    return np.array(enthalpies)  # [J/mol]


def _solid_molar_enthalpies(thermo_data):
    """Integrate each constant solid heat capacity in closed form.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.

    Returns
    -------
    numpy.ndarray
        Species sensible enthalpies ``cp * (TEMPERATURE - DEFAULT_TEMP_REF)``
        [J/mol], shape (num_species,).
    """
    cp_molar = np.array([entry['cp_solid'][0]
                         for entry in thermo_data.values()])  # [J/mol/K]
    return cp_molar * (TEMPERATURE - DEFAULT_TEMP_REF)  # [J/mol]


def _make_liquid(thermo_path):
    """Build the asymmetric liquid mixture at ``TEMPERATURE``.

    Parameters
    ----------
    thermo_path : str
        Path of the JSON property database.

    Returns
    -------
    PharmaPy.Phases.LiquidPhase
        Real liquid phase of mass ``PHASE_MASS`` [kg].
    """
    return LiquidPhase(thermo_path, temp=TEMPERATURE, mass=PHASE_MASS,
                       mass_frac=MASS_FRACTIONS)


def _make_solid(thermo_path):
    """Build the asymmetric solid mixture at ``TEMPERATURE``.

    Parameters
    ----------
    thermo_path : str
        Path of the JSON property database.

    Returns
    -------
    PharmaPy.Phases.SolidPhase
        Real solid phase of mass ``PHASE_MASS`` [kg].
    """
    return SolidPhase(thermo_path, temp=TEMPERATURE, mass=PHASE_MASS,
                      mass_frac=MASS_FRACTIONS)


def _make_vapor(thermo_path):
    """Build the asymmetric vapor mixture at ``TEMPERATURE``.

    Parameters
    ----------
    thermo_path : str
        Path of the JSON property database.

    Returns
    -------
    PharmaPy.Phases.VaporPhase
        Real vapor phase of mass ``PHASE_MASS`` [kg].
    """
    return VaporPhase(thermo_path, temp=TEMPERATURE, mass=PHASE_MASS,
                      mass_frac=MASS_FRACTIONS, verbose=False)


PHASE_FACTORIES = {
    'liquid': _make_liquid,
    'solid': _make_solid,
    'vapor': _make_vapor,
}


# --- Invalid basis --------------------------------------------------------


@pytest.mark.parametrize('total_h', [True, False])
@pytest.mark.parametrize('basis', INVALID_BASES)
@pytest.mark.parametrize('phase_name', ['liquid', 'solid', 'vapor'])
def test_phase_enthalpy_rejects_invalid_basis(thermo_path, phase_name, basis,
                                              total_h):
    """Reject an invalid basis with one message for every provider."""
    phase = PHASE_FACTORIES[phase_name](thermo_path)

    with pytest.raises(ValueError, match=BASIS_MESSAGE):
        phase.getEnthalpy(TEMPERATURE, basis=basis, total_h=total_h)


@pytest.mark.parametrize('total_h', [True, False])
@pytest.mark.parametrize('basis', INVALID_BASES)
@pytest.mark.parametrize('phase_name', ['liquid', 'solid'])
def test_shared_enthalpy_kernel_rejects_invalid_basis(thermo_path, phase_name,
                                                      basis, total_h):
    """Validate at the kernel, which receives both fraction bases."""
    manager = ThermoPhysicalManager(thermo_path)

    with pytest.raises(ValueError, match=BASIS_MESSAGE):
        manager.getEnthalpy(TEMPERATURE, mass_frac=MASS_FRACTIONS,
                            phase=phase_name, basis=basis, total_h=total_h)


@pytest.mark.parametrize('basis', INVALID_BASES)
@pytest.mark.parametrize('phase_name', ['liquid', 'solid'])
def test_phase_density_rejects_invalid_basis(thermo_path, phase_name, basis):
    """Reject an invalid density basis instead of returning molar density."""
    phase = PHASE_FACTORIES[phase_name](thermo_path)

    with pytest.raises(ValueError, match=BASIS_MESSAGE):
        phase.getDensity(basis=basis)


# --- Valid bases are unchanged --------------------------------------------


@pytest.mark.parametrize('phase_name, molar_enthalpies', [
    ('liquid', _liquid_molar_enthalpies),
    ('solid', _solid_molar_enthalpies),
])
def test_valid_basis_enthalpies_match_independent_integrals(
        thermo_path, thermo_data, phase_name, molar_enthalpies):
    """Return mass and molar enthalpies related by the molecular weights."""
    phase = PHASE_FACTORIES[phase_name](thermo_path)
    mw = _mw_kg(thermo_data)  # [kg/mol]

    expected_mole = molar_enthalpies(thermo_data)  # [J/mol]
    expected_mass = expected_mole / mw  # [J/kg]

    species_mole = phase.getEnthalpy(basis='mole', total_h=False)  # [J/mol]
    species_mass = phase.getEnthalpy(basis='mass', total_h=False)  # [J/kg]

    # Species axis in database order, one row for the scalar temperature.
    assert species_mole.shape == (1, len(thermo_data))
    assert species_mass.shape == (1, len(thermo_data))
    np.testing.assert_allclose(species_mole[0], expected_mole, rtol=RTOL)
    np.testing.assert_allclose(species_mass[0], expected_mass, rtol=RTOL)
    # h_mass = h_mole * 1000 / mw, with mw in [g/mol], species by species.
    mw_grams = np.array([entry['mw']
                         for entry in thermo_data.values()])  # [g/mol]
    np.testing.assert_allclose(
        species_mass[0], species_mole[0] * GRAMS_PER_KILOGRAM / mw_grams,
        rtol=RTOL)

    total_mole = phase.getEnthalpy(basis='mole')  # [J/mol]
    total_mass = phase.getEnthalpy(basis='mass')  # [J/kg]

    expected_total_mole = np.dot(_mole_fractions(thermo_data),
                                 expected_mole)  # [J/mol]
    expected_total_mass = np.dot(MASS_FRACTIONS, expected_mass)  # [J/kg]

    assert total_mole == pytest.approx(expected_total_mole, rel=RTOL)
    assert total_mass == pytest.approx(expected_total_mass, rel=RTOL)
    # Mixture totals differ by the mixture molar mass [kg/mol].
    assert total_mass == pytest.approx(
        total_mole / _mixture_molar_mass(thermo_data), rel=RTOL)


@pytest.mark.parametrize('phase_name, density_key', [
    ('liquid', 'rho_liq'),
    ('solid', 'rho_solid'),
])
def test_valid_basis_densities_match_ideal_mixing(
        thermo_path, thermo_data, phase_name, density_key):
    """Return mass and molar ideal-mixing densities on their own bases."""
    phase = PHASE_FACTORIES[phase_name](thermo_path)
    pure_density = np.array([entry[density_key]
                             for entry in thermo_data.values()])  # [kg/m**3]

    expected_mass = 1 / np.sum(MASS_FRACTIONS / pure_density)  # [kg/m**3]
    # [kmol/m**3] = [kg/m**3] / ([kg/mol] * [g/kg]), i.e. [mol/L].
    expected_mole = expected_mass / (
        _mixture_molar_mass(thermo_data) * GRAMS_PER_KILOGRAM)  # [kmol/m**3]

    assert phase.getDensity(basis='mass') == pytest.approx(
        expected_mass, rel=RTOL)
    assert phase.getDensity(basis='mole') == pytest.approx(
        expected_mole, rel=RTOL)
