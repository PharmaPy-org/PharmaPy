"""Test SolidPhase enthalpy-reference precedence and common mixture references.

Issue #332: an omitted ``SolidPhase.getEnthalpy`` reference must honor the
constructor ``temp_ref`` [K], while ``Slurry``, ``Cake``, the slurry
initialization energy balance, and ``Mixer`` must keep evaluating liquid and
solid on one common reference.

The checked-in PFR fixture ``tests/integration/data/pfr_test_pure_comp.json``
supplies synthetic repository values, not measured properties of a real
solid: every species has a constant solid heat capacity [J/mol/K] and a
polynomial liquid heat capacity [J/mol/K]. The solid is pure ``A`` and the
liquid is pure solvent ``solv``, so the two phases have different heat
capacities, molecular weights, and densities. Expected enthalpies are computed
from the JSON data, as the closed-form constant-heat-capacity integral for the
solid and by SciPy quadrature of the liquid polynomial, not from PharmaPy.
Real phase, mixture, and Mixer objects are used; no solver backend is needed.
"""

import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from PharmaPy.Containers import Mixer
from PharmaPy.MixedPhases import Cake, Slurry, energy_balance
from PharmaPy.Phases import LiquidPhase, SolidPhase


pytestmark = pytest.mark.unit

FIXTURE_NAME = 'pfr_test_pure_comp.json'
SOLID_SPECIES = 'A'
LIQUID_SPECIES = 'solv'

GRAMS_PER_KILOGRAM = 1000.0  # [g/kg], exact unit conversion

DEFAULT_TEMP_REF = 298.15  # [K], documented SolidPhase and mixture default
# [K], nondefault solid references: the ice point and a lower arbitrary value.
# Both lie below TEMPERATURE so every sensible enthalpy is nonzero.
ICE_POINT_TEMP_REF = 273.15  # [K]
LOW_TEMP_REF = 250.0  # [K]
TEMPERATURE = 320.0  # [K], above every reference used here

# [K], liquid and solid temperatures of the slurry initialization and Mixer
# cases. Any adiabatic mixing temperature lies between them, which brackets
# the independent root solve.
COLD_TEMP = 300.0  # [K]
HOT_TEMP = 350.0  # [K]

# Rounding allowance for comparisons with the independent integrals [-]. The
# quadrature of a polynomial and the closed-form solid integral are exact to
# floating-point roundoff, and machine-epsilon trace fractions that SolidPhase
# and LiquidPhase substitute for zero entries perturb results near 1e-16.
# Every reference change tested here moves the result by more than 1e-2.
RTOL = 1e-10  # [-]
# Root-solve tolerance of the independent energy balance [K].
TEMP_XTOL = 1e-12  # [K]

# Cake inventories [kg]; unequal so a swapped weighting would be detected.
CAKE_LIQUID_MASS = 0.4  # [kg]
CAKE_SOLID_MASS = 1.0  # [kg]
# Cake size grid [um] and relative bin weights [-] needed only to construct
# the cake porosity; they do not enter the enthalpy weighting.
CAKE_X_DISTRIB = np.array([100.0, 200.0, 300.0])  # [um]
CAKE_DISTRIB = np.array([1.0, 1.0, 1.0])  # [-]

# Slurry fixture: volume [m**3] and a volume-specific number density
# [#/m**3/um] whose third moment by the trapezoidal rule on the 100 um grid is
# 100 * (1e9 * 100**3 + 1.25e8 * 200**3) * 1e-18 = 0.2 [m**3/m**3], so with
# kv = 1 the solid occupies 20 % of the slurry volume.
SLURRY_VOL = 1.0e-3  # [m**3]
SLURRY_X_DISTRIB = np.array([0.0, 100.0, 200.0, 300.0])  # [um]
SLURRY_DISTRIB = np.array([0.0, 1.0e9, 1.25e8, 0.0])  # [#/m**3/um]
SLURRY_SOLID_VOL_FRAC = 0.2  # [-], derived above
KV = 1.0  # [-], cubic shape factor

# LiquidPhase warning for a phase constructed without an inventory.
ZERO_INVENTORY_WARNING = "'mass', 'moles' and 'vol' are all set to zero"


@pytest.fixture(scope='module')
def thermo_path(data_path):
    """Return the PFR pure-component property fixture path.

    Parameters
    ----------
    data_path : dict
        Shared test-data directories.

    Returns
    -------
    str
        Path of the JSON property database.
    """
    return str(data_path['integration'] / FIXTURE_NAME)


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

    # The closed-form solid integral below requires constant heat capacity.
    for entry in data.values():
        assert len(entry['cp_solid']) == 1

    return data


def _pure_fractions(thermo_data, species):
    """Return mass fractions [-] of one pure species in file order.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    species : str
        Name of the pure species.

    Returns
    -------
    numpy.ndarray
        Mass fractions [-], shape (num_species,).
    """
    names = list(thermo_data)
    fractions = np.zeros(len(names))  # [-]
    fractions[names.index(species)] = 1.0
    return fractions


def _mw_kg(thermo_data, species):
    """Return a species molecular weight [kg/mol].

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    species : str
        Species name.

    Returns
    -------
    float
        Molecular weight [kg/mol].
    """
    return thermo_data[species]['mw'] / GRAMS_PER_KILOGRAM  # [kg/mol]


def _solid_molar_enthalpy(thermo_data, species, temp, temp_ref):
    """Integrate a constant solid heat capacity from the reference.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    species : str
        Species name.
    temp : float
        Temperature [K].
    temp_ref : float
        Reference temperature [K].

    Returns
    -------
    float
        Molar sensible enthalpy ``cp * (temp - temp_ref)`` [J/mol].
    """
    cp_molar = thermo_data[species]['cp_solid'][0]  # [J/mol/K]
    return cp_molar * (temp - temp_ref)  # [J/mol]


def _solid_mass_enthalpy(thermo_data, species, temp, temp_ref):
    """Return the constant-heat-capacity solid enthalpy on a mass basis.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    species : str
        Species name.
    temp : float
        Temperature [K].
    temp_ref : float
        Reference temperature [K].

    Returns
    -------
    float
        Mass sensible enthalpy [J/kg].
    """
    return (_solid_molar_enthalpy(thermo_data, species, temp, temp_ref)
            / _mw_kg(thermo_data, species))  # [J/kg]


def _liquid_mass_enthalpy(thermo_data, species, temp, temp_ref):
    """Integrate the liquid heat-capacity polynomial by quadrature.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    species : str
        Species name.
    temp : float
        Temperature [K].
    temp_ref : float
        Reference temperature [K].

    Returns
    -------
    float
        Mass sensible enthalpy [J/kg], from ascending coefficients
        ``cp = sum_k c_k T**k`` [J/mol/K].
    """
    coefficients = np.asarray(thermo_data[species]['cp_liq'])  # ascending

    def cp_molar(temperature):
        """Return liquid heat capacity [J/mol/K] at ``temperature`` [K]."""
        return np.polynomial.polynomial.polyval(temperature, coefficients)

    molar_enthalpy, _ = quad(cp_molar, temp_ref, temp)  # [J/mol]
    return molar_enthalpy / _mw_kg(thermo_data, species)  # [J/kg]


def _make_solid(thermo_path, thermo_data, temp_ref, **kwargs):
    """Build a pure-A solid, optionally with a constructor reference.

    Parameters
    ----------
    thermo_path : str
        Path of the JSON property database.
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    temp_ref : float or None
        Constructor enthalpy reference [K]; ``None`` uses the default.
    **kwargs
        Additional ``SolidPhase`` keywords, with units as documented there.

    Returns
    -------
    PharmaPy.Phases.SolidPhase
        Real solid phase.
    """
    if temp_ref is not None:
        kwargs['temp_ref'] = temp_ref
    kwargs.setdefault('temp', TEMPERATURE)
    return SolidPhase(thermo_path,
                      mass_frac=_pure_fractions(thermo_data, SOLID_SPECIES),
                      **kwargs)


# --- Direct SolidPhase calls ---------------------------------------------


@pytest.mark.parametrize('temp_ref', [DEFAULT_TEMP_REF, ICE_POINT_TEMP_REF,
                                      LOW_TEMP_REF])
def test_omitted_reference_uses_constructor_reference(
        thermo_path, thermo_data, temp_ref):
    """Honor the stored solid reference on mass and molar bases."""
    solid = _make_solid(thermo_path, thermo_data, temp_ref, mass=1.0)

    expected_mass = _solid_mass_enthalpy(
        thermo_data, SOLID_SPECIES, TEMPERATURE, temp_ref)  # [J/kg]
    expected_mole = _solid_molar_enthalpy(
        thermo_data, SOLID_SPECIES, TEMPERATURE, temp_ref)  # [J/mol]

    assert solid.getEnthalpy(basis='mass') == pytest.approx(
        expected_mass, rel=RTOL)
    assert solid.getEnthalpy(basis='mole') == pytest.approx(
        expected_mole, rel=RTOL)


def test_default_constructor_reference_is_unchanged(thermo_path, thermo_data):
    """Keep the 298.15 K result for a solid built without ``temp_ref``."""
    solid = _make_solid(thermo_path, thermo_data, None, mass=1.0)

    expected = _solid_mass_enthalpy(
        thermo_data, SOLID_SPECIES, TEMPERATURE, DEFAULT_TEMP_REF)  # [J/kg]

    assert solid.temp_ref == DEFAULT_TEMP_REF
    assert solid.getEnthalpy() == pytest.approx(expected, rel=RTOL)


@pytest.mark.parametrize('explicit_ref', [DEFAULT_TEMP_REF, LOW_TEMP_REF])
def test_explicit_reference_overrides_stored_reference(
        thermo_path, thermo_data, explicit_ref):
    """Let an explicit method reference, even 298.15 K, take precedence."""
    solid = _make_solid(thermo_path, thermo_data, ICE_POINT_TEMP_REF, mass=1.0)

    expected_mass = _solid_mass_enthalpy(
        thermo_data, SOLID_SPECIES, TEMPERATURE, explicit_ref)  # [J/kg]
    expected_mole = _solid_molar_enthalpy(
        thermo_data, SOLID_SPECIES, TEMPERATURE, explicit_ref)  # [J/mol]

    assert solid.getEnthalpy(temp_ref=explicit_ref) == pytest.approx(
        expected_mass, rel=RTOL)
    assert solid.getEnthalpy(temp_ref=explicit_ref, basis='mole') == \
        pytest.approx(expected_mole, rel=RTOL)
    assert solid.temp_ref == ICE_POINT_TEMP_REF


def test_species_enthalpies_honor_constructor_reference(
        thermo_path, thermo_data):
    """Apply the stored reference to every species with ``total_h=False``."""
    solid = _make_solid(thermo_path, thermo_data, ICE_POINT_TEMP_REF, mass=1.0)

    expected = np.array([
        _solid_mass_enthalpy(thermo_data, name, TEMPERATURE,
                             ICE_POINT_TEMP_REF)
        for name in thermo_data])[np.newaxis, :]  # [J/kg], (1, num_species)

    species_enthalpy = solid.getEnthalpy(total_h=False)  # [J/kg]

    assert species_enthalpy.shape == expected.shape
    np.testing.assert_allclose(species_enthalpy, expected, rtol=RTOL)


def test_temperature_profile_honors_constructor_reference(
        thermo_path, thermo_data):
    """Apply the stored reference along a temperature profile axis."""
    solid = _make_solid(thermo_path, thermo_data, LOW_TEMP_REF, mass=1.0)
    temps = np.array([COLD_TEMP, TEMPERATURE, HOT_TEMP])  # [K]

    expected = np.array([
        _solid_mass_enthalpy(thermo_data, SOLID_SPECIES, temp, LOW_TEMP_REF)
        for temp in temps])  # [J/kg]

    np.testing.assert_allclose(solid.getEnthalpy(temp=temps), expected,
                               rtol=RTOL)


# --- Cake -----------------------------------------------------------------


def _make_cake(thermo_path, thermo_data, solid_temp_ref):
    """Attach a liquid and a referenced solid to a Cake at TEMPERATURE.

    Parameters
    ----------
    thermo_path : str
        Path of the JSON property database.
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    solid_temp_ref : float
        Constructor enthalpy reference of the solid [K].

    Returns
    -------
    PharmaPy.MixedPhases.Cake
        Cake with liquid mass ``CAKE_LIQUID_MASS`` and solid mass
        ``CAKE_SOLID_MASS`` [kg].
    """
    liquid = LiquidPhase(
        thermo_path, temp=TEMPERATURE, mass=CAKE_LIQUID_MASS,
        mass_frac=_pure_fractions(thermo_data, LIQUID_SPECIES))
    solid = _make_solid(thermo_path, thermo_data, solid_temp_ref,
                        mass=CAKE_SOLID_MASS, x_distrib=CAKE_X_DISTRIB,
                        distrib=CAKE_DISTRIB)
    cake = Cake()
    cake.Phases = [liquid, solid]

    # Fixture self-check: the weights below are the attached inventories.
    assert cake.Liquid_1.mass == pytest.approx(CAKE_LIQUID_MASS, rel=RTOL)
    assert cake.Solid_1.mass == pytest.approx(CAKE_SOLID_MASS, rel=RTOL)
    return cake


def _cake_expected(thermo_data, temp_ref):
    """Return the mass-weighted cake enthalpy on one common reference.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    temp_ref : float
        Common reference temperature [K].

    Returns
    -------
    float
        Cake enthalpy [J/kg of liquid plus solid].
    """
    h_liquid = _liquid_mass_enthalpy(
        thermo_data, LIQUID_SPECIES, TEMPERATURE, temp_ref)  # [J/kg]
    h_solid = _solid_mass_enthalpy(
        thermo_data, SOLID_SPECIES, TEMPERATURE, temp_ref)  # [J/kg]
    return ((CAKE_LIQUID_MASS * h_liquid + CAKE_SOLID_MASS * h_solid)
            / (CAKE_LIQUID_MASS + CAKE_SOLID_MASS))  # [J/kg]


@pytest.mark.parametrize('solid_temp_ref', [DEFAULT_TEMP_REF,
                                            ICE_POINT_TEMP_REF])
def test_cake_default_uses_common_reference(
        thermo_path, thermo_data, solid_temp_ref):
    """Weight both cake phases on 298.15 K regardless of the solid state."""
    cake = _make_cake(thermo_path, thermo_data, solid_temp_ref)

    expected = _cake_expected(thermo_data, DEFAULT_TEMP_REF)  # [J/kg]

    assert cake.getEnthalpy() == pytest.approx(expected, rel=RTOL)


@pytest.mark.parametrize('common_ref', [ICE_POINT_TEMP_REF, LOW_TEMP_REF])
def test_cake_explicit_reference_applies_to_both_phases(
        thermo_path, thermo_data, common_ref):
    """Evaluate liquid and solid on one explicitly requested reference."""
    cake = _make_cake(thermo_path, thermo_data, ICE_POINT_TEMP_REF)

    expected = _cake_expected(thermo_data, common_ref)  # [J/kg]

    assert cake.getEnthalpy(temp_ref=common_ref) == pytest.approx(
        expected, rel=RTOL)


# --- Slurry ---------------------------------------------------------------


def _make_slurry(thermo_path, thermo_data, solid_temp_ref,
                 liquid_temp=TEMPERATURE, solid_temp=TEMPERATURE):
    """Attach a liquid and a referenced solid to a Slurry.

    Parameters
    ----------
    thermo_path : str
        Path of the JSON property database.
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    solid_temp_ref : float
        Constructor enthalpy reference of the solid [K].
    liquid_temp : float, optional
        Liquid temperature before mixing [K].
    solid_temp : float, optional
        Solid temperature before mixing [K].

    Returns
    -------
    slurry : PharmaPy.MixedPhases.Slurry
        Slurry of volume ``SLURRY_VOL`` [m**3].
    masses : tuple of float
        Independently derived liquid and solid masses [kg].
    """
    # The Slurry setter assigns the liquid inventory, so the phase starts
    # empty and LiquidPhase reports that documented zero-inventory state.
    with pytest.warns(RuntimeWarning, match=ZERO_INVENTORY_WARNING):
        liquid = LiquidPhase(
            thermo_path, temp=liquid_temp,
            mass_frac=_pure_fractions(thermo_data, LIQUID_SPECIES))
    solid = _make_solid(thermo_path, thermo_data, solid_temp_ref,
                        temp=solid_temp, kv=KV)
    slurry = Slurry(vol=SLURRY_VOL, x_distrib=SLURRY_X_DISTRIB,
                    distrib=SLURRY_DISTRIB)
    slurry.Phases = (liquid, solid)

    mass_liquid = ((1 - SLURRY_SOLID_VOL_FRAC) * SLURRY_VOL
                   * thermo_data[LIQUID_SPECIES]['rho_liq'])  # [kg]
    mass_solid = (SLURRY_SOLID_VOL_FRAC * SLURRY_VOL
                  * thermo_data[SOLID_SPECIES]['rho_solid'])  # [kg]

    # Fixture self-check against the independently derived inventories.
    assert slurry.Liquid_1.mass == pytest.approx(mass_liquid, rel=RTOL)
    assert slurry.Solid_1.mass == pytest.approx(mass_solid, rel=RTOL)
    return slurry, (mass_liquid, mass_solid)


def _phase_enthalpies(thermo_data, temp, temp_ref):
    """Return liquid and solid mass enthalpies on one reference.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    temp : float
        Temperature [K].
    temp_ref : float
        Common reference temperature [K].

    Returns
    -------
    numpy.ndarray
        ``[liquid, solid]`` enthalpies [J/kg].
    """
    return np.array([
        _liquid_mass_enthalpy(thermo_data, LIQUID_SPECIES, temp, temp_ref),
        _solid_mass_enthalpy(thermo_data, SOLID_SPECIES, temp, temp_ref),
    ])  # [J/kg]


@pytest.mark.parametrize('common_ref', [None, ICE_POINT_TEMP_REF,
                                        LOW_TEMP_REF])
def test_slurry_mass_enthalpy_uses_common_reference(
        thermo_path, thermo_data, common_ref):
    """Weight both slurry phases by mass on the default or explicit reference."""
    slurry, masses = _make_slurry(thermo_path, thermo_data,
                                  ICE_POINT_TEMP_REF)
    kwargs = {} if common_ref is None else {'temp_ref': common_ref}
    expected_ref = DEFAULT_TEMP_REF if common_ref is None else common_ref  # [K]

    masses = np.array(masses)  # [kg]
    expected = (np.dot(masses, _phase_enthalpies(
        thermo_data, TEMPERATURE, expected_ref)) / masses.sum())  # [J/kg]

    assert slurry.getEnthalpy(TEMPERATURE, volumetric=False, **kwargs) == \
        pytest.approx(expected, rel=RTOL)


@pytest.mark.parametrize('common_ref', [None, ICE_POINT_TEMP_REF])
def test_slurry_volumetric_enthalpy_uses_common_reference(
        thermo_path, thermo_data, common_ref):
    """Weight phases by volume fraction and density, with positional inputs."""
    slurry, _ = _make_slurry(thermo_path, thermo_data, ICE_POINT_TEMP_REF)
    kwargs = {} if common_ref is None else {'temp_ref': common_ref}
    expected_ref = DEFAULT_TEMP_REF if common_ref is None else common_ref  # [K]

    vol_fracs = np.array([1 - SLURRY_SOLID_VOL_FRAC,
                          SLURRY_SOLID_VOL_FRAC])  # [-], [liquid, solid]
    densities = np.array([thermo_data[LIQUID_SPECIES]['rho_liq'],
                          thermo_data[SOLID_SPECIES]['rho_solid']])  # [kg/m**3]
    expected = np.sum(vol_fracs * densities * _phase_enthalpies(
        thermo_data, TEMPERATURE, expected_ref))  # [J/m**3]

    # Positional volume fractions and densities, as Crystallizers pass them.
    assert slurry.getEnthalpy(TEMPERATURE, vol_fracs, densities, **kwargs) == \
        pytest.approx(expected, rel=RTOL)
    assert slurry.getEnthalpy(TEMPERATURE, **kwargs) == pytest.approx(
        expected, rel=RTOL)


def _mixing_root(thermo_data, inlet_terms, masses_out,
                 temp_ref=DEFAULT_TEMP_REF):
    """Solve an independent adiabatic mixing balance on one reference.

    Parameters
    ----------
    thermo_data : dict
        Raw JSON property entries keyed by species name.
    inlet_terms : list of tuple
        ``(mass_liquid, mass_solid, temp)`` per inlet, in [kg], [kg], [K].
    masses_out : tuple of float
        Total liquid and solid masses [kg].
    temp_ref : float, optional
        Common reference temperature of every phase enthalpy [K]; default
        298.15 K. Mass conservation makes the root independent of it.

    Returns
    -------
    float
        Mixture temperature [K].
    """
    h_in = sum(
        np.dot([mass_liq, mass_sol],
               _phase_enthalpies(thermo_data, temp, temp_ref))
        for mass_liq, mass_sol, temp in inlet_terms)  # [J]

    def residual(temp):
        """Return inlet minus outlet enthalpy [J] at ``temp`` [K]."""
        return h_in - np.dot(masses_out, _phase_enthalpies(
            thermo_data, temp, temp_ref))  # [J]

    return brentq(residual, COLD_TEMP, HOT_TEMP, xtol=TEMP_XTOL)  # [K]


def test_slurry_initialization_temperature_uses_common_reference(
        thermo_path, thermo_data):
    """Initialize a mixed-temperature slurry from one common reference."""
    slurry, masses = _make_slurry(
        thermo_path, thermo_data, ICE_POINT_TEMP_REF,
        liquid_temp=COLD_TEMP, solid_temp=HOT_TEMP)
    default_slurry, _ = _make_slurry(
        thermo_path, thermo_data, DEFAULT_TEMP_REF,
        liquid_temp=COLD_TEMP, solid_temp=HOT_TEMP)

    mass_liquid, mass_solid = masses  # [kg]
    expected = _mixing_root(
        thermo_data,
        [(mass_liquid, 0.0, COLD_TEMP), (0.0, mass_solid, HOT_TEMP)],
        masses)  # [K]

    assert slurry.temp == pytest.approx(expected, rel=RTOL)
    assert slurry.temp == pytest.approx(default_slurry.temp, rel=RTOL)


@pytest.mark.parametrize('common_ref', [ICE_POINT_TEMP_REF, LOW_TEMP_REF])
def test_energy_balance_helper_forwards_common_reference(
        thermo_path, thermo_data, common_ref):
    """Evaluate phase and mixture enthalpies on the requested reference."""
    slurry, masses = _make_slurry(
        thermo_path, thermo_data, ICE_POINT_TEMP_REF,
        liquid_temp=COLD_TEMP, solid_temp=HOT_TEMP)
    # Self-check: the setter leaves the phase temperatures unmixed, so the
    # helper below solves a nontrivial balance.
    assert slurry.Liquid_1.temp == COLD_TEMP
    assert slurry.Solid_1.temp == HOT_TEMP

    mass_liquid, mass_solid = masses  # [kg]
    expected = _mixing_root(
        thermo_data,
        [(mass_liquid, 0.0, COLD_TEMP), (0.0, mass_solid, HOT_TEMP)],
        masses, temp_ref=common_ref)  # [K]

    temp_mix = energy_balance(slurry, 'mass', temp_ref=common_ref)  # [K]

    assert temp_mix == pytest.approx(expected, rel=RTOL)
    assert temp_mix == pytest.approx(slurry.temp, rel=RTOL)


# --- Mixer ----------------------------------------------------------------


def test_mixer_outlet_temperature_ignores_solid_constructor_reference(
        thermo_path, thermo_data):
    """Mix a liquid with a nondefault-reference slurry through solve_unit."""
    liquid_mass_in = 1.0  # [kg], solids-free inlet
    outlet_temps = []  # [K]
    for solid_temp_ref in (DEFAULT_TEMP_REF, ICE_POINT_TEMP_REF):
        liquid_inlet = LiquidPhase(
            thermo_path, temp=COLD_TEMP, mass=liquid_mass_in,
            mass_frac=_pure_fractions(thermo_data, LIQUID_SPECIES))
        slurry_inlet, (mass_liq_slurry, mass_sol_slurry) = _make_slurry(
            thermo_path, thermo_data, solid_temp_ref,
            liquid_temp=HOT_TEMP, solid_temp=HOT_TEMP)

        mixer = Mixer()
        # The solids-free inlet is first because Mixer reads name_species
        # from Inlets[0], which Slurry does not define.
        mixer.Inlets = [liquid_inlet, slurry_inlet]
        mixer.solve_unit()

        # Fixture self-check: the outlet holds the combined inventory.
        assert mixer.Outlet.Liquid_1.mass == pytest.approx(
            liquid_mass_in + mass_liq_slurry, rel=RTOL)
        assert mixer.Outlet.Solid_1.mass == pytest.approx(
            mass_sol_slurry, rel=RTOL)
        outlet_temps.append(mixer.Outlet.temp)

    masses_out = (liquid_mass_in + mass_liq_slurry, mass_sol_slurry)  # [kg]
    expected = _mixing_root(
        thermo_data,
        [(liquid_mass_in, 0.0, COLD_TEMP),
         (mass_liq_slurry, mass_sol_slurry, HOT_TEMP)],
        masses_out)  # [K]

    assert outlet_temps[1] == pytest.approx(expected, rel=RTOL)
    assert outlet_temps[1] == pytest.approx(outlet_temps[0], rel=RTOL)
