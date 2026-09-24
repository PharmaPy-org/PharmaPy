"""Shape contract of ``VaporPhase.getHeatVaporization`` and its Drying consumer.

Regression coverage for issue #101. The mass-basis branch used to reassign
``deltahvap`` to its subcritical column subset, so a phase holding any species
that is supercritical at every requested temperature returned fewer columns on
a mass basis than on a mole basis. Mixture weighting in
``VaporPhase.getEnthalpy`` then failed, and ``Drying.energy_balance`` only
worked while that accidental narrowing happened to match its volatile count.

Expected latent heats are derived here with a scalar loop over one temperature
at a time, independent of the vectorised two-dimensional fancy indexing under
test, which is where the defect lived. The phase fixture carries three species
of which two stay subcritical at the probe temperatures, so the converted
column axis has length two and a reversed or mis-assigned column is caught by
value rather than only by shape. The Drying cases use real ``LiquidPhase`` and
``VaporPhase`` collaborators from that same documented synthetic database; a
nonzero carrier drying rate verifies that its real zero latent-heat column does
not disturb volatile pairing.
"""

import json

import numpy as np
import pytest

from PharmaPy.Drying_Model import Drying
from PharmaPy.Phases import LiquidPhase, VaporPhase

pytestmark = pytest.mark.unit


# Synthetic three-species pure-component table, in the JSON layout that
# ThermoPhysicalManager reads. Every value here is a construction-only test
# assumption chosen to exercise the supercritical branch; none is taken from a
# measured or published dataset for a real substance.
#
# The critical temperatures straddle the probe temperatures, so "light" is
# supercritical there while "heavy" and "medium" still condense. Molar masses
# are kept distinct so that a mis-assigned species column changes the value
# rather than cancelling out.
#
# Schema, one entry per species, with the unit of every field:
#
# ``mw``         : molar mass [g/mol].
# ``t_crit``     : critical temperature [K]; a species is treated as
#                  supercritical, and so non-condensable, above it.
# ``rho_liq``    : saturated liquid density [kg/m**3]. Required by the
#                  constructor; not exercised by these tests.
# ``cp_liq``     : polynomial coefficients of the liquid molar heat capacity
#                  [J/mol/K], ordered from the constant term upward, so that
#                  ``cp = sum_i c_i * T**i`` with ``T`` in [K]. A single
#                  coefficient is a temperature-independent heat capacity.
# ``cp_vapor``   : same polynomial form for the vapor molar heat capacity
#                  [J/mol/K]; used for species above ``t_crit``.
# ``p_vap``      : Antoine coefficients ``[A, B, C]`` for
#                  ``log10(p) = A - B / (T + C)`` as implemented in
#                  ``VaporPhase.AntoineEquation``, giving ``p`` in [Pa] with
#                  ``T`` in [K]; ``A`` is [-], ``B`` is [K], ``C`` is [K].
#                  Required by the constructor; no test here evaluates a vapor
#                  pressure.
# ``delta_hvap`` : latent heat of vaporization [J/mol], tabulated at
#                  ``tref_hvap`` and extrapolated by the Watson correlation.
# ``tref_hvap``  : reference temperature of ``delta_hvap`` [K].
THERMO_THREE_SPECIES = {
    "light": {
        "mw": 18.0,
        "t_crit": 650.0,
        "rho_liq": 1000.0,
        "cp_liq": [75.0],
        "cp_vapor": [35.0],
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
        "cp_vapor": [60.0],
        "p_vap": [8.0, 1650.0, -40.0],
        "delta_hvap": 50000.0,
        "tref_hvap": 350.0,
    },
}

MOLAR_MASS = np.array([18.0, 100.0, 46.0])  # [g/mol]
T_CRIT = np.array([650.0, 700.0, 720.0])  # [K]
DELTA_HVAP_REF = np.array([40000.0, 60000.0, 50000.0])  # [J/mol] at TREF_HVAP
TREF_HVAP = np.array([350.0, 350.0, 350.0])  # [K]
NUM_SPECIES = len(MOLAR_MASS)  # [-]

# Exponent of the Watson correlation used by VaporPhase.getHeatVaporization to
# extrapolate the tabulated latent heat away from its reference temperature.
WATSON_EXPONENT = 0.38  # [-]

G_PER_KG = 1000.0  # [g/kg], molar mass is tabulated in g/mol

# Every value lies above t_crit of "light" and below t_crit of "heavy", so one
# species is supercritical and two are subcritical. Four temperatures against
# three species against two subcritical species keeps all three axis lengths
# different, so a transposed or mis-sliced result cannot match by coincidence.
SUPERCRIT_TEMPS = np.array([660.0, 665.0, 670.0, 675.0])  # [K]


def _latent_heat_mole_reference(temp_scalar):
    """Latent heat of every species at one temperature, on a molar basis.

    Scalar reference implementation of the Watson correlation, written as a
    loop over species so that it shares no indexing logic with the vectorised
    implementation under test.

    Parameters
    ----------
    temp_scalar : float
        Temperature at which the latent heat is evaluated [K].

    Returns
    -------
    ndarray
        Latent heat of each species [J/mol]. Entries for species that are
        supercritical at ``temp_scalar`` are zero, because a supercritical
        species cannot condense.
    """
    latent_heat = np.zeros(NUM_SPECIES)  # [J/mol]
    for species in range(NUM_SPECIES):
        if temp_scalar < T_CRIT[species]:
            reduced = ((T_CRIT[species] - temp_scalar)
                       / (T_CRIT[species] - TREF_HVAP[species]))  # [-]
            latent_heat[species] = (DELTA_HVAP_REF[species]
                                    * reduced**WATSON_EXPONENT)

    return latent_heat


@pytest.fixture
def vapor_phase(tmp_path):
    """Three-species ``VaporPhase`` built from the straddling-t_crit fixture."""
    path = tmp_path / "thermo_three_species.json"
    path.write_text(json.dumps(THERMO_THREE_SPECIES))

    return VaporPhase(str(path), temp=350.0, moles=1.0,
                      mole_frac=np.full(NUM_SPECIES, 1.0 / NUM_SPECIES),
                      check_input=False)


def test_multitemperature_mass_basis_keeps_supercritical_columns(vapor_phase):
    """A supercritical species keeps a zero column instead of being dropped."""
    latent_mass = vapor_phase.getHeatVaporization(SUPERCRIT_TEMPS, basis="mass")
    latent_mole = vapor_phase.getHeatVaporization(SUPERCRIT_TEMPS, basis="mole")

    assert latent_mass.shape == (len(SUPERCRIT_TEMPS), NUM_SPECIES)
    assert latent_mass.shape == latent_mole.shape

    expected_mole = np.vstack(
        [_latent_heat_mole_reference(t) for t in SUPERCRIT_TEMPS])  # [J/mol]
    expected_mass = expected_mole / MOLAR_MASS * G_PER_KG  # [J/kg]

    np.testing.assert_allclose(latent_mole, expected_mole, rtol=1e-12)
    np.testing.assert_allclose(latent_mass, expected_mass, rtol=1e-12)

    # "light" is supercritical at every probe temperature and cannot condense.
    np.testing.assert_array_equal(latent_mass[:, 0],
                                  np.zeros(len(SUPERCRIT_TEMPS)))
    assert np.all(latent_mass[:, 1:] > 0.0)


def test_single_temperature_mass_basis_keeps_supercritical_columns(vapor_phase):
    """The scalar-temperature branch preserves the species axis as well."""
    temp = 660.0  # [K], above t_crit of "light" only

    latent_mass = vapor_phase.getHeatVaporization(temp, basis="mass")

    assert latent_mass.shape == (NUM_SPECIES,)

    expected_mass = (_latent_heat_mole_reference(temp)
                     / MOLAR_MASS * G_PER_KG)  # [J/kg]
    np.testing.assert_allclose(latent_mass, expected_mass, rtol=1e-12)
    assert latent_mass[0] == 0.0


def test_getenthalpy_mass_basis_weights_full_species_axis(vapor_phase):
    """Mixture weighting no longer fails when a species is supercritical.

    This is the failure reported in issue #101: the mass-basis latent heat
    reached ``np.dot`` with fewer columns than the mass-fraction vector had
    entries. The temperature count has to equal the species count here,
    because ``getEnthalpy`` separately compares ``temp`` against ``t_crit``
    elementwise and cannot yet accept a differing number of temperatures.
    """
    temps = np.array([660.0, 665.0, 670.0])  # [K], one per species

    enthalpy_mass = vapor_phase.getEnthalpy(temp=temps, basis="mass")  # [J/kg]
    enthalpy_mole = vapor_phase.getEnthalpy(temp=temps, basis="mole")  # [J/mol]

    assert enthalpy_mass.shape == (len(temps),)
    assert np.all(np.isfinite(enthalpy_mass))

    # A correctly shaped but wrongly weighted mixture would still be finite, so
    # pin the value: the two bases must describe the same mixture, related only
    # by its average molar mass. The single-temperature sibling test in
    # test_phases_vapor_enthalpy.py asserts this same relation.
    np.testing.assert_allclose(
        enthalpy_mass, enthalpy_mole * G_PER_KG / vapor_phase.mw_av,
        rtol=1e-12)

    # The latent contribution specifically must be weighted by mass fraction on
    # a mass basis and by mole fraction on a mole basis.
    latent_mass = vapor_phase.getHeatVaporization(temps, basis="mass")  # [J/kg]
    latent_mole = vapor_phase.getHeatVaporization(temps, basis="mole")  # [J/mol]
    np.testing.assert_allclose(
        latent_mass.dot(vapor_phase.mass_frac),
        latent_mole.dot(vapor_phase.mole_frac) * G_PER_KG / vapor_phase.mw_av,
        rtol=1e-12)


@pytest.mark.parametrize("temps", [
    np.array([400.0, 420.0, 440.0, 460.0]),  # [K], every species subcritical
    np.array([400.0]),  # [K], single-element array
])
def test_all_subcritical_mass_basis_is_unchanged(vapor_phase, temps):
    """The previously working all-subcritical path keeps its exact values.

    With no supercritical species the old expression indexed every column, so
    the in-place conversion must reproduce it exactly rather than merely
    closely.
    """
    latent_mass = vapor_phase.getHeatVaporization(temps, basis="mass")
    latent_mole = vapor_phase.getHeatVaporization(temps, basis="mole")

    # Pre-fix expression, evaluated on the same molar intermediate.
    every_species = np.arange(NUM_SPECIES)
    previous = (np.atleast_2d(latent_mole)[:, every_species]
                / MOLAR_MASS[every_species] * G_PER_KG)  # [J/kg]

    np.testing.assert_array_equal(np.atleast_2d(latent_mass), previous)
    assert np.all(latent_mass > 0.0)


def _real_drying_balance(vapor_phase, num_nodes, temp_cond, dry_rate):
    """Evaluate real Drying energy collaborators and derive latent reference.

    Parameters
    ----------
    vapor_phase : VaporPhase
        Real three-species vapor collaborator.
    num_nodes : int
        Axial node count [-].
    temp_cond : ndarray
        Condensed temperatures [K].
    dry_rate : ndarray
        Full-width component mass drying rates [kg/m**3/s].

    Returns
    -------
    tuple of ndarray
        Actual and independently assembled condensed temperature rates [K/s].
    """
    dryer = Drying(number_nodes=num_nodes, supercrit_names=["light"])
    dryer.idx_volatiles = np.array([1, 2])  # species indices [-]
    dryer.idx_supercrit = np.array([0])  # species indices [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_sol = 1000.0  # [kg/m**3]
    dryer.cp_sol = 1000.0  # [J/kg/K]
    dryer.rho_liq = np.full(num_nodes, 800.0)  # [kg/m**3]
    dryer.h_T_j = 0.0  # [W/m**2/K], isolates the latent term
    dryer.a_V = 1.0  # [m**2/m**3]
    dryer.dz = np.ones(num_nodes)  # [m]
    dryer.Vapor_1 = vapor_phase
    dryer.Liquid_1 = LiquidPhase(
        vapor_phase.path_data,
        temp=660.0,  # [K]
        mass=1.0,  # [kg]
        mass_frac=np.array([0.0, 0.4, 0.6]),  # [-]
    )

    saturation = np.full(num_nodes, 0.5)  # [-]
    x_liq = np.tile(np.array([0.4, 0.6]), (num_nodes, 1))  # [-]
    _, actual_rate = dryer.energy_balance(
        time=0.0,
        temp_gas=np.full(num_nodes, 295.0),  # [K], wet-bulb reference
        temp_sol=temp_cond,
        satur=saturation,
        y_gas=np.tile(np.array([0.8, 0.1, 0.1]), (num_nodes, 1)),  # [-]
        x_liq=x_liq,
        u_gas=np.zeros(num_nodes),  # [m/s]
        rho_gas=np.ones(num_nodes),  # [kg/m**3]
        dry_rate=dry_rate,
        inputs={"temp": 295.0},  # [K]
    )

    latent_heat = np.atleast_2d(
        vapor_phase.getHeatVaporization(temp_cond, basis="mass")
    )  # [J/kg]
    np.testing.assert_array_equal(latent_heat[:, 0], np.zeros(num_nodes))
    liquid_cp = dryer.Liquid_1.getCp(
        temp_cond,
        mass_frac=np.column_stack((x_liq, np.zeros(num_nodes))),
        basis="mass",
    )  # [J/kg/K]
    heat_capacity = (
        dryer.rho_sol * (1.0 - dryer.porosity) * dryer.cp_sol
        + dryer.porosity * saturation * liquid_cp * dryer.rho_liq
    )  # [J/m**3/K]
    latent_power = (
        dry_rate[:, dryer.idx_volatiles]
        * latent_heat[:, dryer.idx_volatiles]
    ).sum(axis=1)  # [J/m**3/s]
    expected_rate = -latent_power / heat_capacity  # [K/s]
    return actual_rate, expected_rate


def test_energy_balance_uses_only_real_volatile_latent_heat_columns(
        vapor_phase):
    """Pair each volatile rate with the same real latent-heat column."""
    dry_rate = np.array([
        [0.5, 0.036, 0.184],
        [0.5, 0.054, 0.230],
        [0.5, 0.018, 0.092],
        [0.5, 0.027, 0.046],
    ])  # [kg/m**3/s], supercritical carrier is deliberately nonzero

    actual_rate, expected_rate = _real_drying_balance(
        vapor_phase, 4, SUPERCRIT_TEMPS.copy(), dry_rate
    )

    np.testing.assert_allclose(actual_rate, expected_rate)


def test_energy_balance_handles_real_single_node_latent_heat(vapor_phase):
    """Normalize a real one-node 1-D latent vector before column selection."""
    dry_rate = np.array([[0.5, 0.036, 0.184]])  # [kg/m**3/s]
    temp_cond = np.array([660.0])  # [K]
    latent_heat = vapor_phase.getHeatVaporization(
        temp_cond, basis="mass"
    )  # [J/kg]
    assert latent_heat.shape == (NUM_SPECIES,)

    actual_rate, expected_rate = _real_drying_balance(
        vapor_phase, 1, temp_cond, dry_rate
    )

    assert actual_rate.shape == (1,)
    np.testing.assert_allclose(actual_rate, expected_rate)
