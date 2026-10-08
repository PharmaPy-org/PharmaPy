"""Temperature-axis contract of ``VaporPhase`` enthalpy and latent heat.

Regression coverage for issue #178. ``VaporPhase.getEnthalpy`` compared an
array of temperatures elementwise against the per-species critical
temperatures, so it raised unless the temperature count happened to equal the
species count, and then classified each species with one mask for the whole
call. A temperature exactly equal to a critical temperature fell out of both
strict subsets, and the mixture latent heat used ``np.dot(deltaVap, frac)``,
a matrix product that mixes rows for a paired temperature/composition profile.
The latent-heat provider raised for temperature sets straddling a critical
temperature.

The contract under test classifies every ``(temperature, species)`` entry
separately: strictly above ``t_crit`` is supercritical, everything else,
including ``T == t_crit``, is subcritical. Every row of a multi-temperature
result must equal the evaluation at that row's temperature alone (and its
composition row, for a paired profile). A species without critical data,
whose ``t_crit`` is parsed as NaN, is non-condensing: liquid sensible heat and
zero latent heat at every temperature. For every other species evaluated at or
below its critical temperature, non-finite Watson inputs, an invalid
reference temperature, or an overflowing Watson ratio raise ``ValueError``,
and so do fractions without one entry per species.

Expected values come from two independent sources: scalar-temperature calls of
the public methods, and a scalar reference written here from the closed-form
integrals of the fixture's at most linear heat-capacity polynomials and the
Watson correlation, sharing no indexing with the implementation. One test
spells out that hand derivation term by term at a critical temperature. The
synthetic three-species table below, and deliberate variants of it (a
nonvolatile species without critical data, missing or infinite Watson inputs),
are written to ``tmp_path``; no solver backend is used and every test is
fast.
"""

import copy
import json
import warnings

import numpy as np
import pytest

from PharmaPy.Phases import VaporPhase

pytestmark = pytest.mark.unit


# Synthetic pure-component table in the JSON layout read by
# ThermoPhysicalManager. Every value is a construction-only test assumption,
# not a measured dataset. Critical temperatures 650, 700 and 720 K let the
# probe temperatures make zero, one, two or three species supercritical.
# Heat capacities are polynomials ``cp = c0 + c1 * T`` [J/mol/K] with ``T``
# in [K]; ``light`` has a linear liquid and vapor cp and ``medium`` a linear
# vapor cp, so the sensible integral is not merely cp * dT, and the padded
# second coefficient of the constant polynomials is zero. Molar masses
# [g/mol], latent heats ``delta_hvap`` [J/mol] and their reference
# temperatures ``tref_hvap`` [K] are distinct per species so a misplaced
# column changes the value. ``rho_liq`` [kg/m**3] and the Antoine
# coefficients ``p_vap`` [A: -, B: K, C: K] satisfy the constructor only.
PURE_COMPONENTS = {
    "light": {
        "mw": 18.0,
        "t_crit": 650.0,
        "rho_liq": 1000.0,
        "cp_liq": [60.0, 0.05],
        "cp_vapor": [30.0, 0.01],
        "p_vap": [8.0, 1500.0, -40.0],
        "delta_hvap": 40000.0,
        "tref_hvap": 350.0,
    },
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
    "medium": {
        "mw": 46.0,
        "t_crit": 720.0,
        "rho_liq": 850.0,
        "cp_liq": [110.0],
        "cp_vapor": [60.0, 0.02],
        "p_vap": [8.0, 1650.0, -40.0],
        "delta_hvap": 50000.0,
        "tref_hvap": 400.0,
    },
}

MOLAR_MASS = np.array([18.0, 100.0, 46.0])  # [g/mol]
T_CRIT = np.array([650.0, 700.0, 720.0])  # [K]
CP_LIQ = np.array([[60.0, 0.05], [150.0, 0.0], [110.0, 0.0]])  # [J/mol/K], [J/mol/K**2]
CP_VAPOR = np.array([[30.0, 0.01], [80.0, 0.0], [60.0, 0.02]])  # [J/mol/K], [J/mol/K**2]
DELTA_HVAP_REF = np.array([40000.0, 60000.0, 50000.0])  # [J/mol] at TREF_HVAP
TREF_HVAP = np.array([350.0, 350.0, 400.0])  # [K]
NUM_SPECIES = len(MOLAR_MASS)  # [-]

# Phase composition; unequal fractions so a permuted weighting is detected.
PHASE_MOLE_FRAC = np.array([0.2, 0.5, 0.3])  # [-]

TEMP_REF = 298.15  # [K], default liquid reference of getEnthalpy
# Correlation exponent documented by VaporPhase.getHeatVaporization.
WATSON_EXPONENT = 0.38  # [-]
G_PER_KG = 1000.0  # [g/kg], exact unit conversion

# Same computation per row as in a scalar call; only summation order and
# fraction conversion can differ, by a few ulps.
ROW_RTOL = 1e-14  # [-]
# Closed-form reference versus the implementation's polynomial evaluation.
REFERENCE_RTOL = 1e-12  # [-]

# Temperature sets [K] with counts below, equal to, and above the species
# count. The equal_asymmetric set has light and heavy supercritical in row 0,
# no species in row 1, and light only in row 2, so neither an elementwise
# temperature/species pairing nor a transposed result can match. The unsorted
# equal_subcritical set keeps every species subcritical, which the former
# elementwise comparison accepted; it isolates the row mixing of the former
# matrix-product mixture weighting from criticality classification. The
# equal_exact_critical set puts 700 K, heavy's t_crit, at heavy's own index
# while light is supercritical in every row and nothing straddles: the former
# elementwise split then silently dropped heavy's sensible heat from every
# mixture row instead of raising. The straddling set crosses the critical
# temperatures of light and heavy, and the exact_critical set places each
# species at its own critical temperature in turn.
TEMPERATURE_SETS = {
    "fewer": np.array([660.0, 705.0]),
    "equal_asymmetric": np.array([705.0, 400.0, 690.0]),
    "equal_subcritical": np.array([520.0, 400.0, 450.0]),
    "equal_exact_critical": np.array([660.0, 700.0, 690.0]),
    "more": np.array([400.0, 655.0, 705.0, 725.0, 690.0]),
    "straddling": np.array([400.0, 705.0, 660.0]),
    "exact_critical": np.array([650.0, 700.0, 720.0]),
}

# Asymmetric composition rows [-], each summing to one; the first
# num_temperatures rows pair with a temperature set.
PROFILE_ROWS = np.array([
    [0.1, 0.3, 0.6],
    [0.5, 0.2, 0.3],
    [0.25, 0.7, 0.05],
    [0.6, 0.1, 0.3],
    [0.15, 0.45, 0.4],
])  # [-]


def _write_phase(tmp_path, components=None):
    """Write a component table and build a vapor phase from it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the JSON table.
    components : dict, optional
        Component table in the PharmaPy JSON layout; ``PURE_COMPONENTS`` by
        default.

    Returns
    -------
    VaporPhase
        One mole [mol] of vapor at 350 K with ``PHASE_MOLE_FRAC``.
    """
    table = PURE_COMPONENTS if components is None else components
    path = tmp_path / "thermo_temperature_axis.json"
    path.write_text(json.dumps(table))
    return VaporPhase(str(path), temp=350.0, moles=1.0,
                      mole_frac=PHASE_MOLE_FRAC, check_input=False)


@pytest.fixture
def vapor_phase(tmp_path):
    """Three-species vapor phase built from ``PURE_COMPONENTS``."""
    return _write_phase(tmp_path)


def _linear_cp_integral(coefficients, temp):
    """Integrate ``cp = c0 + c1 * T`` from ``TEMP_REF`` to ``temp``.

    Parameters
    ----------
    coefficients : ndarray
        ``[c0, c1]`` in [J/mol/K] and [J/mol/K**2].
    temp : float
        Upper integration temperature [K].

    Returns
    -------
    float
        ``c0 * (T - Tr) + c1 / 2 * (T**2 - Tr**2)`` [J/mol].
    """
    const, slope = coefficients
    return (const * (temp - TEMP_REF)
            + slope / 2 * (temp**2 - TEMP_REF**2))


def _reference_latent_mole(temp):
    """Watson latent heat of every species at one temperature.

    Parameters
    ----------
    temp : float
        Temperature [K].

    Returns
    -------
    ndarray
        Latent heat [J/mol], shape ``(NUM_SPECIES,)``; zero where ``temp``
        lies strictly above the species critical temperature.
    """
    latent = np.zeros(NUM_SPECIES)  # [J/mol]
    for species in range(NUM_SPECIES):
        if not temp > T_CRIT[species]:
            reduced = ((T_CRIT[species] - temp)
                       / (T_CRIT[species] - TREF_HVAP[species]))  # [-]
            latent[species] = DELTA_HVAP_REF[species] * reduced**WATSON_EXPONENT
    return latent


def _reference_species(temp, basis):
    """Species enthalpy at one temperature from closed-form integrals.

    Parameters
    ----------
    temp : float
        Temperature [K].
    basis : {'mass', 'mole'}
        Basis of the returned values.

    Returns
    -------
    ndarray
        Species enthalpy relative to liquid at ``TEMP_REF``, shape
        ``(NUM_SPECIES,)``, in [J/kg] or [J/mol] by ``basis``.
    """
    latent = _reference_latent_mole(temp)  # [J/mol]
    enthalpy = np.empty(NUM_SPECIES)  # [J/mol]
    for species in range(NUM_SPECIES):
        if temp > T_CRIT[species]:
            enthalpy[species] = _linear_cp_integral(CP_VAPOR[species], temp)
        else:
            enthalpy[species] = (_linear_cp_integral(CP_LIQ[species], temp)
                                 + latent[species])
    if basis == "mass":
        return enthalpy * G_PER_KG / MOLAR_MASS  # [J/kg]
    return enthalpy


def _to_mass_frac(mole_frac):
    """Convert mole fractions to mass fractions row by row.

    Parameters
    ----------
    mole_frac : ndarray
        Mole fractions [-], species on the last axis.

    Returns
    -------
    ndarray
        Mass fractions [-] with the input shape.
    """
    weighted = mole_frac * MOLAR_MASS  # [g/mol]
    return weighted / weighted.sum(axis=-1, keepdims=True)


def _to_mole_frac(mass_frac):
    """Convert mass fractions to mole fractions row by row.

    Parameters
    ----------
    mass_frac : ndarray
        Mass fractions [-], species on the last axis.

    Returns
    -------
    ndarray
        Mole fractions [-] with the input shape.
    """
    moles_per_mass = mass_frac / MOLAR_MASS  # [mol/g]
    return moles_per_mass / moles_per_mass.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("name", list(TEMPERATURE_SETS))
def test_species_enthalpy_rows_match_scalar_evaluations(
        vapor_phase, name, basis):
    """Each species row equals its scalar-temperature evaluation."""
    temps = TEMPERATURE_SETS[name]  # [K]

    observed = vapor_phase.getEnthalpy(
        temp=temps, total_h=False, basis=basis)  # [J/kg] or [J/mol]

    assert observed.shape == (len(temps), NUM_SPECIES)
    scalar_rows = np.vstack([
        vapor_phase.getEnthalpy(temp=temp, total_h=False, basis=basis)[0]
        for temp in temps])  # [J/kg] or [J/mol]
    reference = np.vstack([_reference_species(temp, basis)
                           for temp in temps])  # [J/kg] or [J/mol]
    np.testing.assert_allclose(observed, scalar_rows, rtol=ROW_RTOL)
    np.testing.assert_allclose(observed, reference, rtol=REFERENCE_RTOL)


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("name", list(TEMPERATURE_SETS))
def test_mixture_enthalpy_rows_match_scalar_evaluations(
        vapor_phase, name, basis):
    """A fixed phase composition weights each temperature row separately."""
    temps = TEMPERATURE_SETS[name]  # [K]
    frac = (_to_mass_frac(PHASE_MOLE_FRAC) if basis == "mass"
            else PHASE_MOLE_FRAC)  # [-]

    observed = vapor_phase.getEnthalpy(temp=temps, basis=basis)  # [J/kg] or [J/mol]

    assert observed.shape == (len(temps),)
    scalar_totals = np.array([vapor_phase.getEnthalpy(temp=temp, basis=basis)
                              for temp in temps])  # [J/kg] or [J/mol]
    reference = np.array([_reference_species(temp, basis) @ frac
                          for temp in temps])  # [J/kg] or [J/mol]
    np.testing.assert_allclose(observed, scalar_totals, rtol=ROW_RTOL)
    np.testing.assert_allclose(observed, reference, rtol=REFERENCE_RTOL)


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("name", list(TEMPERATURE_SETS))
def test_latent_heat_rows_match_scalar_evaluations(vapor_phase, name, basis):
    """Latent heat is classified per entry and never warns.

    The straddling and exact-critical sets used to raise on a negative or
    undefined Watson ratio; now every supercritical entry is zero, every
    other entry is the Watson value, and no fractional power of a negative
    number is attempted.
    """
    temps = TEMPERATURE_SETS[name]  # [K]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed = vapor_phase.getHeatVaporization(temps, basis=basis)  # [J/kg] or [J/mol]

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []
    assert observed.shape == (len(temps), NUM_SPECIES)
    scalar_rows = np.vstack([vapor_phase.getHeatVaporization(temp, basis=basis)
                             for temp in temps])  # [J/kg] or [J/mol]
    reference = np.vstack([_reference_latent_mole(temp)
                           for temp in temps])  # [J/mol]
    if basis == "mass":
        reference = reference * G_PER_KG / MOLAR_MASS  # [J/kg]
    np.testing.assert_allclose(observed, scalar_rows, rtol=ROW_RTOL)
    np.testing.assert_allclose(observed, reference, rtol=REFERENCE_RTOL)
    supercritical = temps[:, np.newaxis] > T_CRIT  # [-]
    assert np.all(observed[supercritical] == 0.0)
    assert np.all(observed[~supercritical] >= 0.0)


def test_straddling_set_mixes_classifications_within_one_column(vapor_phase):
    """One species is supercritical in some rows and subcritical in others."""
    temps = TEMPERATURE_SETS["straddling"]  # [K], 400, 705 and 660 K

    latent = vapor_phase.getHeatVaporization(temps, basis="mole")  # [J/mol]
    species = vapor_phase.getEnthalpy(temp=temps, total_h=False,
                                      basis="mole")  # [J/mol]

    # light (Tc 650 K): subcritical at 400 K only.
    assert latent[0, 0] > 0.0
    assert latent[1, 0] == latent[2, 0] == 0.0
    # heavy (Tc 700 K): supercritical at 705 K only.
    assert latent[0, 1] > 0.0 and latent[2, 1] > 0.0
    assert latent[1, 1] == 0.0
    np.testing.assert_allclose(
        species[1, 1], _linear_cp_integral(CP_VAPOR[1], 705.0),
        rtol=REFERENCE_RTOL)
    np.testing.assert_allclose(
        species[2, 1],
        _linear_cp_integral(CP_LIQ[1], 660.0) + latent[2, 1],
        rtol=REFERENCE_RTOL)


EXACT_CRITICAL_TEMP = 700.0  # [K], t_crit of heavy; light is supercritical


def _hand_derived_exact_critical(basis):
    """Species enthalpy at 700 K, derived term by term.

    Relative to liquid at Tr = 298.15 K, molar basis:

    - light (Tc 650 K, supercritical): vapor sensible heat only,
      30 (700 - Tr) + 0.01 / 2 (700**2 - Tr**2) [J/mol].
    - heavy (Tc 700 K, T == Tc, subcritical): liquid sensible heat
      150 (700 - Tr) plus a Watson latent heat of exactly zero.
    - medium (Tc 720 K, subcritical): 110 (700 - Tr) plus
      50000 ((720 - 700) / (720 - 400))**0.38 [J/mol].

    Parameters
    ----------
    basis : {'mass', 'mole'}
        Basis of the returned values.

    Returns
    -------
    expected : ndarray
        Species enthalpy, shape ``(NUM_SPECIES,)``, [J/kg] or [J/mol].
    frac : ndarray
        Phase fractions on the same basis [-].
    """
    temp = EXACT_CRITICAL_TEMP  # [K]
    span = temp - TEMP_REF  # [K]
    expected_mole = np.array([
        CP_VAPOR[0, 0] * span + CP_VAPOR[0, 1] / 2 * (temp**2 - TEMP_REF**2),
        CP_LIQ[1, 0] * span,
        CP_LIQ[2, 0] * span + DELTA_HVAP_REF[2] * (
            (T_CRIT[2] - temp) / (T_CRIT[2] - TREF_HVAP[2]))**WATSON_EXPONENT,
    ])  # [J/mol]
    if basis == "mass":
        return (expected_mole * G_PER_KG / MOLAR_MASS,
                _to_mass_frac(PHASE_MOLE_FRAC))  # [J/kg], [-]
    return expected_mole, PHASE_MOLE_FRAC  # [J/mol], [-]


@pytest.mark.parametrize("basis", ["mass", "mole"])
def test_exact_critical_scalar_mixture_hand_derivation(vapor_phase, basis):
    """The mixture keeps heavy's sensible heat at its own Tc.

    Earlier code silently dropped heavy's liquid sensible heat from this
    total while returning a finite value.
    """
    expected, frac = _hand_derived_exact_critical(basis)

    total = vapor_phase.getEnthalpy(temp=EXACT_CRITICAL_TEMP,
                                    basis=basis)  # [J/kg] or [J/mol]

    assert total == pytest.approx(expected @ frac, rel=REFERENCE_RTOL)
    assert np.ndim(total) == 0


@pytest.mark.parametrize("basis", ["mass", "mole"])
def test_exact_critical_scalar_species_hand_derivation(vapor_phase, basis):
    """No species column is lost when one species sits exactly at Tc."""
    expected, _ = _hand_derived_exact_critical(basis)

    species = vapor_phase.getEnthalpy(temp=EXACT_CRITICAL_TEMP,
                                      total_h=False, basis=basis)
    latent = vapor_phase.getHeatVaporization(EXACT_CRITICAL_TEMP,
                                             basis=basis)

    assert species.shape == (1, NUM_SPECIES)
    np.testing.assert_allclose(species[0], expected, rtol=REFERENCE_RTOL)
    # 0**0.38 is exactly zero, so heavy's latent heat is exactly zero.
    assert latent[1] == 0.0
    assert latent[0] == 0.0
    assert latent[2] > 0.0


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("temp", [400.0, 660.0, np.array([660.0])],
                         ids=["all_subcritical", "mixed", "one_element"])
def test_scalar_path_values_and_shapes(vapor_phase, temp, basis):
    """Scalar and one-element temperatures keep their established contract."""
    temp_scalar = float(np.atleast_1d(temp)[0])  # [K]
    expected = _reference_species(temp_scalar, basis)  # [J/kg] or [J/mol]
    frac = (_to_mass_frac(PHASE_MOLE_FRAC) if basis == "mass"
            else PHASE_MOLE_FRAC)  # [-]
    latent = _reference_latent_mole(temp_scalar)  # [J/mol]
    if basis == "mass":
        latent = latent * G_PER_KG / MOLAR_MASS  # [J/kg]

    species = vapor_phase.getEnthalpy(temp=temp, total_h=False, basis=basis)
    total = vapor_phase.getEnthalpy(temp=temp, basis=basis)
    observed_latent = vapor_phase.getHeatVaporization(temp, basis=basis)

    assert species.shape == (1, NUM_SPECIES)
    np.testing.assert_allclose(species[0], expected, rtol=REFERENCE_RTOL)
    assert np.ndim(total) == 0
    assert total == pytest.approx(expected @ frac, rel=REFERENCE_RTOL)
    assert observed_latent.shape == (NUM_SPECIES,)
    np.testing.assert_allclose(observed_latent, latent, rtol=REFERENCE_RTOL)


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("keyword", ["mass_frac", "mole_frac"])
@pytest.mark.parametrize("composition", ["paired", "fixed"])
@pytest.mark.parametrize("name", ["fewer", "equal_asymmetric",
                                  "equal_subcritical", "equal_exact_critical",
                                  "more"])
def test_composition_rows_match_scalar_evaluations(
        vapor_phase, name, composition, keyword, basis):
    """Each mixture row equals the scalar-temperature, scalar-composition call.

    For the equal-count cases the former ``np.dot(deltaVap, frac)`` was a
    matrix product that returned a plausible square array mixing unrelated
    temperature and composition rows.
    """
    temps = TEMPERATURE_SETS[name]  # [K]
    num_temp = len(temps)  # [-]
    supplied = (PROFILE_ROWS[:num_temp] if composition == "paired"
                else PROFILE_ROWS[0])  # [-], basis given by keyword
    rows = np.broadcast_to(supplied, (num_temp, NUM_SPECIES))  # [-]

    observed = vapor_phase.getEnthalpy(
        temp=temps, basis=basis, **{keyword: supplied})  # [J/kg] or [J/mol]

    assert observed.shape == (num_temp,)
    scalar_totals = np.array([
        vapor_phase.getEnthalpy(temp=temp, basis=basis,
                                **{keyword: rows[ind].copy()})
        for ind, temp in enumerate(temps)])  # [J/kg] or [J/mol]
    np.testing.assert_allclose(observed, scalar_totals, rtol=ROW_RTOL)

    if keyword == "mass_frac":
        weights = rows if basis == "mass" else _to_mole_frac(rows)  # [-]
    else:
        weights = _to_mass_frac(rows) if basis == "mass" else rows  # [-]
    reference = np.array([
        _reference_species(temp, basis) @ weights[ind]
        for ind, temp in enumerate(temps)])  # [J/kg] or [J/mol]
    np.testing.assert_allclose(observed, reference, rtol=REFERENCE_RTOL)


def test_one_row_profile_with_scalar_temperature(vapor_phase):
    """A scalar temperature accepts a one-row profile and returns a scalar."""
    temp = 660.0  # [K]

    observed = vapor_phase.getEnthalpy(
        temp=temp, mole_frac=PROFILE_ROWS[:1], basis="mole")  # [J/mol]

    assert np.ndim(observed) == 0
    assert observed == pytest.approx(
        _reference_species(temp, "mole") @ PROFILE_ROWS[0],
        rel=REFERENCE_RTOL)


@pytest.mark.parametrize("temp, profile", [
    (np.array([660.0, 705.0]), PROFILE_ROWS[:3]),
    (np.array([660.0, 705.0, 400.0, 690.0]), PROFILE_ROWS[:3]),
    (660.0, PROFILE_ROWS[:2]),
], ids=["fewer_temperatures", "more_temperatures", "scalar_temperature"])
@pytest.mark.parametrize("keyword", ["mass_frac", "mole_frac"])
def test_profile_row_count_mismatch_raises(vapor_phase, temp, profile,
                                           keyword):
    """A profile must have one row per requested temperature."""
    with pytest.raises(ValueError, match="paired row by row with temp"):
        vapor_phase.getEnthalpy(temp=temp, **{keyword: profile})


@pytest.mark.parametrize("tref_hvap", [650.0, 660.0],
                         ids=["equal_to_tcrit", "above_tcrit"])
def test_invalid_watson_reference_raises_with_species_name(tmp_path,
                                                           tref_hvap):
    """A subcritical species needs ``tref_hvap < t_crit``.

    At equality the old code returned an infinite or NaN latent heat. The
    species is only checked where the Watson branch actually uses it: at
    temperatures where light is supercritical everywhere, its zero column is
    returned without error.
    """
    components = copy.deepcopy(PURE_COMPONENTS)
    components["light"]["tref_hvap"] = tref_hvap  # [K]
    phase = _write_phase(tmp_path, components)

    with pytest.raises(ValueError, match=r"tref_hvap < t_crit.*\['light'\]"):
        phase.getHeatVaporization(400.0, basis="mole")
    with pytest.raises(ValueError, match=r"tref_hvap < t_crit.*\['light'\]"):
        phase.getEnthalpy(temp=np.array([660.0, 400.0]), basis="mole")

    temps = np.array([660.0, 690.0])  # [K], light supercritical in every row
    latent = phase.getHeatVaporization(temps, basis="mole")  # [J/mol]
    np.testing.assert_array_equal(latent[:, 0], np.zeros(len(temps)))
    np.testing.assert_allclose(
        latent, np.vstack([_reference_latent_mole(temp) for temp in temps]),
        rtol=REFERENCE_RTOL)


@pytest.mark.parametrize("temp", [
    np.nan, np.inf, np.array([400.0, np.nan]), np.array([-np.inf, 400.0]),
], ids=["nan", "inf", "nan_in_array", "neg_inf_in_array"])
@pytest.mark.parametrize("method", ["getHeatVaporization", "getEnthalpy"])
def test_non_finite_temperature_raises(vapor_phase, temp, method):
    """A NaN or infinite temperature is reported as such, not as a ratio."""
    with pytest.raises(ValueError, match=r"temp must be finite \[K\]"):
        getattr(vapor_phase, method)(temp)


@pytest.mark.parametrize("temp, message", [
    (np.full((2, 2), 400.0), "one-dimensional"),
    (np.array([]), "at least one temperature"),
], ids=["two_dimensional", "empty"])
@pytest.mark.parametrize("method", ["getHeatVaporization", "getEnthalpy"])
def test_unsupported_temperature_shape_raises(vapor_phase, temp, message,
                                              method):
    """Only a scalar or a non-empty one-dimensional array is accepted."""
    with pytest.raises(ValueError, match=message):
        getattr(vapor_phase, method)(temp)


def test_missing_vapor_cp_is_needed_only_for_supercritical_entries(tmp_path):
    """Without ``cp_vapor`` data, subcritical temperatures still evaluate."""
    components = copy.deepcopy(PURE_COMPONENTS)
    for properties in components.values():
        del properties["cp_vapor"]
    phase = _write_phase(tmp_path, components)
    temps = np.array([400.0, 450.0, 500.0, 550.0])  # [K], all subcritical

    observed = phase.getEnthalpy(temp=temps, total_h=False,
                                 basis="mole")  # [J/mol]

    np.testing.assert_allclose(
        observed, np.vstack([_reference_species(temp, "mole")
                             for temp in temps]),
        rtol=REFERENCE_RTOL)
    with pytest.raises(AttributeError, match=r"\['light'\].*'cp_vapor'"):
        phase.getEnthalpy(temp=np.array([400.0, 660.0]), basis="mole")


@pytest.mark.parametrize("supplied", [
    {"mole_frac": [0.5, 0.5]},
    {"mass_frac": np.full((2, 2), 0.5)},
    {"mole_frac": np.full((1, 2, 3), 1.0 / 3.0)},
], ids=["short_list", "narrow_profile", "three_dimensional"])
@pytest.mark.parametrize("total_h", [True, False])
def test_wrong_width_composition_raises_before_conversion(vapor_phase,
                                                          supplied, total_h):
    """Fractions need one entry per species on the last axis."""
    name = next(iter(supplied))

    with pytest.raises(ValueError,
                       match=rf"{name} must have shape \(num_species,\)"):
        vapor_phase.getEnthalpy(temp=400.0, total_h=total_h, **supplied)


# Nonvolatile species in the layout of a real API entry: liquid, vapor and
# Antoine data but no critical or latent-heat data, so ParseDatabase stores
# NaN for its t_crit, tref_hvap and delta_hvap. Construction-only values:
# mw [g/mol], rho_liq [kg/m**3], cp_liq and cp_vapor [J/mol/K, J/mol/K**2],
# p_vap [A: -, B: K, C: K].
NONVOLATILE_API = {
    "mw": 250.0,
    "rho_liq": 1200.0,
    "cp_liq": [300.0, 0.2],
    "cp_vapor": [200.0],
    "p_vap": [8.0, 3000.0, -40.0],
}
API_CP_LIQ = np.array([300.0, 0.2])  # [J/mol/K], [J/mol/K**2]
API_MOLAR_MASS = 250.0  # [g/mol]
WITH_API_MOLE_FRAC = np.array([0.2, 0.3, 0.35, 0.15])  # [-], api last


def _phase_with_api(tmp_path):
    """Build the three fixture species plus the nonvolatile ``api``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the JSON table.

    Returns
    -------
    VaporPhase
        One mole [mol] of vapor at 350 K with ``WITH_API_MOLE_FRAC``.
    """
    components = copy.deepcopy(PURE_COMPONENTS)
    components["api"] = copy.deepcopy(NONVOLATILE_API)
    path = tmp_path / "thermo_with_api.json"
    path.write_text(json.dumps(components))
    return VaporPhase(str(path), temp=350.0, moles=1.0,
                      mole_frac=WITH_API_MOLE_FRAC, check_input=False)


def _reference_with_api(temp, basis):
    """Species enthalpy with the non-condensing ``api`` appended.

    Parameters
    ----------
    temp : float
        Temperature [K].
    basis : {'mass', 'mole'}
        Basis of the returned values.

    Returns
    -------
    ndarray
        Enthalpy of light, heavy, medium and api, shape ``(4,)``, in [J/kg]
        or [J/mol]. The api contributes liquid sensible heat and no latent
        heat at every temperature.
    """
    api_mole = _linear_cp_integral(API_CP_LIQ, temp)  # [J/mol]
    if basis == "mass":
        api = api_mole * G_PER_KG / API_MOLAR_MASS  # [J/kg]
    else:
        api = api_mole  # [J/mol]
    return np.append(_reference_species(temp, basis), api)


API_TEMPERATURES = [
    400.0, 660.0, np.array([400.0, 660.0, 705.0, 690.0, 725.0]),
]  # [K]
API_TEMPERATURE_IDS = ["scalar_all_subcritical", "scalar_mixed", "array"]


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("temp", API_TEMPERATURES, ids=API_TEMPERATURE_IDS)
def test_species_without_critical_data_mixture(tmp_path, temp, basis):
    """A NaN ``t_crit`` species contributes liquid sensible heat to the mix.

    At 400 K this is the earlier all-subcritical result. At 660 K, where
    light is supercritical, the earlier mixed branch silently dropped the
    api's sensible heat from this total; the array covers every row at once.
    """
    phase = _phase_with_api(tmp_path)
    temps = np.atleast_1d(temp)  # [K]
    reference = np.vstack([_reference_with_api(value, basis)
                           for value in temps])  # [J/kg] or [J/mol]
    mole_frac = WITH_API_MOLE_FRAC  # [-]
    weights = (mole_frac * np.append(MOLAR_MASS, API_MOLAR_MASS)
               if basis == "mass" else mole_frac)  # [g/mol] or [-]
    weights = weights / weights.sum()  # [-], on the requested basis

    total = phase.getEnthalpy(temp=temp, basis=basis)  # [J/kg] or [J/mol]

    np.testing.assert_allclose(np.atleast_1d(total), reference @ weights,
                               rtol=REFERENCE_RTOL)
    assert np.ndim(total) == np.ndim(temp)


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("temp", API_TEMPERATURES, ids=API_TEMPERATURE_IDS)
def test_species_without_critical_data_species_and_latent(tmp_path, temp,
                                                          basis):
    """A NaN ``t_crit`` species keeps its column and has zero latent heat."""
    phase = _phase_with_api(tmp_path)
    temps = np.atleast_1d(temp)  # [K]
    reference = np.vstack([_reference_with_api(value, basis)
                           for value in temps])  # [J/kg] or [J/mol]

    species = phase.getEnthalpy(temp=temp, total_h=False, basis=basis)
    latent = np.atleast_2d(phase.getHeatVaporization(temp, basis=basis))

    assert species.shape == reference.shape
    np.testing.assert_allclose(species, reference, rtol=REFERENCE_RTOL)
    np.testing.assert_array_equal(latent[:, -1], np.zeros(len(temps)))


@pytest.mark.parametrize("species, field, value", [
    ("heavy", "t_crit", float("inf")),
    ("heavy", "tref_hvap", None),
    ("medium", "delta_hvap", None),
], ids=["infinite_tcrit", "missing_tref", "missing_delta_hvap"])
@pytest.mark.parametrize("method", ["getHeatVaporization", "getEnthalpy"])
def test_invalid_watson_data_with_critical_data_raises(
        tmp_path, species, field, value, method):
    """Watson inputs of a species with critical data must be finite.

    ``None`` removes the field, so ParseDatabase stores NaN for it; an
    infinite ``t_crit`` is written as JSON ``Infinity``. Either would make
    the Watson latent heat NaN at 400 K, where every species is subcritical.
    """
    components = copy.deepcopy(PURE_COMPONENTS)
    if value is None:
        del components[species][field]
    else:
        components[species][field] = value  # [K]
    phase = _write_phase(tmp_path, components)

    with pytest.raises(ValueError,
                       match=rf"needs finite t_crit.*\['{species}'\]"):
        getattr(phase, method)(400.0, basis="mole")


def test_overflowing_watson_ratio_raises(tmp_path):
    """The defensive ratio check rejects an arithmetic overflow.

    The data pass the reference checks (finite, tref_hvap < t_crit), but
    t_crit - temp overflows to infinity for an extreme, finite temperature,
    so the Watson ratio would be infinite.
    """
    components = copy.deepcopy(PURE_COMPONENTS)
    largest = np.finfo(float).max  # [K]
    components["light"]["t_crit"] = largest  # [K]
    phase = _write_phase(tmp_path, components)

    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match=r"Watson ratio.*\['light'\]"):
            phase.getHeatVaporization(-largest, basis="mole")


def _without_fields(*fields):
    """Copy ``PURE_COMPONENTS`` with fields removed from every species.

    Parameters
    ----------
    *fields : str
        Property names to omit for every species, so ParseDatabase creates
        no attribute for them.

    Returns
    -------
    dict
        Component table in the PharmaPy JSON layout.
    """
    components = copy.deepcopy(PURE_COMPONENTS)
    for properties in components.values():
        for field in fields:
            del properties[field]
    return components


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("temp", [730.0, np.array([725.0, 760.0, 740.0])],
                         ids=["scalar", "array"])
def test_no_latent_heat_data_needed_above_every_tcrit(tmp_path, temp, basis):
    """Without any delta_hvap or tref_hvap, supercritical entries evaluate.

    Every temperature lies above every t_crit (720 K at most), so no entry
    uses the Watson correlation: latent heat is zero and the enthalpy is the
    closed-form vapor sensible heat.
    """
    phase = _write_phase(tmp_path, _without_fields("delta_hvap", "tref_hvap"))
    temps = np.atleast_1d(temp)  # [K]
    expected_mole = np.vstack([
        [_linear_cp_integral(CP_VAPOR[species], value)
         for species in range(NUM_SPECIES)]
        for value in temps])  # [J/mol]
    if basis == "mass":
        expected = expected_mole * G_PER_KG / MOLAR_MASS  # [J/kg]
        frac = _to_mass_frac(PHASE_MOLE_FRAC)  # [-]
    else:
        expected = expected_mole  # [J/mol]
        frac = PHASE_MOLE_FRAC  # [-]

    latent = phase.getHeatVaporization(temp, basis=basis)  # [J/kg] or [J/mol]
    species = phase.getEnthalpy(temp=temp, total_h=False, basis=basis)
    total = phase.getEnthalpy(temp=temp, basis=basis)  # [J/kg] or [J/mol]

    expected_latent_shape = ((NUM_SPECIES,) if np.ndim(temp) == 0
                             else (len(temps), NUM_SPECIES))
    assert latent.shape == expected_latent_shape
    np.testing.assert_array_equal(latent, np.zeros(expected_latent_shape))
    np.testing.assert_allclose(species, expected, rtol=REFERENCE_RTOL)
    np.testing.assert_allclose(np.atleast_1d(total), expected @ frac,
                               rtol=REFERENCE_RTOL)


@pytest.mark.parametrize("field", ["tref_hvap", "delta_hvap"])
@pytest.mark.parametrize("method", ["getHeatVaporization", "getEnthalpy"])
def test_absent_watson_field_raises_species_error(tmp_path, field, method):
    """A field every species omits is reported as missing Watson data.

    At 400 K every species is subcritical and uses the Watson correlation,
    so the specific error must name all three species.
    """
    phase = _write_phase(tmp_path, _without_fields(field))

    with pytest.raises(
            ValueError,
            match=r"needs finite t_crit.*\['light', 'heavy', 'medium'\]"):
        getattr(phase, method)(400.0, basis="mole")
