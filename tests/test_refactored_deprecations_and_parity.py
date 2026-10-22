"""Deprecation contracts and Gate G7 parity for refactored phase, stream, and vessel APIs.

Verifies that:
1. Deprecated legacy shims in ``Phases_Refactored``, ``Streams_Refactored``,
   and ``MultiPhaseVessel`` emit ``DeprecationWarning`` attributed to the
   caller while preserving their compatibility behavior.
2. Earlier fixes from Gate G7 (#18 / #178 vapor enthalpy and latent heat,
   #32 supersaturation ratio deprecation, #156 solid/liquid/vapor mole
   accounting, #332 enthalpy reference temperature, and #414 zero-sentinel
   heat-capacity handling) hold on the refactored phase and stream classes.
3. Legacy bridge conversions (``to_legacy`` and ``from_legacy``) preserve
   non-default pressure [Pa] and reference temperature [K].
4. ``to_stream()`` / ``to_phase()`` round-trips and nested ``MixedStream``
   flattening preserve class, family, ordering, and extensive properties across
   liquid, solid, and vapor families with unequal inventories.
"""

import json
import os

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.MixedPhases_Refactored import MixedPhase, MixedStream
from PharmaPy.Phases import (
    LiquidPhase as LegacyLiquidPhase,
    SolidPhase as LegacySolidPhase,
    VaporPhase as LegacyVaporPhase,
)
from PharmaPy.Phases_Refactored import (
    LiquidPhase,
    SolidPhase,
    VaporPhase,
    classify_phases as classify_phases_phase,
    getPropsPhaseMix as get_props_phase_mix,
)
from PharmaPy.Reactors_Refactored import BatchReactor
from PharmaPy.Streams_Refactored import (
    BatchToFlowConnector,
    Interpolation,
    LiquidStream,
    SolidStream,
    VaporStream,
    classify_phases as classify_phases_stream,
    getPropsPhaseMix as get_props_stream_mix,
)

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

# Composition over the 5-species flowsheet database (A, B, C, D, solvent); sum = 1.0 [-].
LIQUID_MASS_FRAC = np.array([0.15, 0.20, 0.05, 0.0, 0.60])  # [-]
ALT_MOLE_FRAC = np.array([0.25, 0.25, 0.10, 0.0, 0.40])  # [-]
ALT_MASS_FRAC = np.array([0.30, 0.10, 0.20, 0.0, 0.40])  # [-]
SOLID_MASS_FRAC = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # [-]
VAPOR_MOLE_FRAC = np.array([0.30, 0.20, 0.0, 0.0, 0.50])  # [-]

# Representative thermodynamic states and inventories for unit verification.
SAMPLE_TEMP = 320.0  # [K]
DEFAULT_TEMP_REF = 298.15  # [K], standard reference temperature
CUSTOM_TEMP_REF = 305.0  # [K], non-default reference temperature (#332)
SAMPLE_PRES = 101325.0  # [Pa], standard atmospheric pressure
ELEVATED_PRES = 300000.0  # [Pa], non-default pressure for ideal-gas bridge check
SAMPLE_MASS = 2.5  # [kg]
SOLID_MASS = 1.2  # [kg], distinct from SAMPLE_MASS to avoid symmetric inventories
SAMPLE_FLOW = 0.75  # [kg/s]
GRAMS_PER_KG = 1000.0  # [g/kg], exact SI conversion factor
# Double-precision relative tolerance for identical algebraic evaluations.
PARITY_RTOL = 1e-12  # [-]


class _PhaseHolder:
    """Minimal container for testing legacy classify_phases helpers."""

    def __init__(self, phases):
        self.Phases = phases


def test_classify_phases_emits_deprecation_and_indexes_vapor_correctly():
    """Both classify_phases shims warn at caller stacklevel and number Vapor phases."""
    liquid = LiquidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mass_frac=LIQUID_MASS_FRAC
    )
    solid = SolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC
    )
    vapor = VaporPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mole_frac=VAPOR_MOLE_FRAC
    )

    for helper in (classify_phases_phase, classify_phases_stream):
        holder = _PhaseHolder([liquid, solid, vapor])
        with pytest.warns(DeprecationWarning, match="classify_phases") as record:
            helper(holder)
        assert record[0].filename == __file__
        assert holder.Liquid_1 is liquid
        assert holder.Solid_1 is solid
        assert holder.Vapor_1 is vapor
        assert vapor.name == "Vapor_1"

        named_holder = _PhaseHolder([liquid, solid])
        with pytest.warns(DeprecationWarning, match="classify_phases") as record_named:
            helper(named_holder, names=("feed_liq", "seed_sol"))
        assert record_named[0].filename == __file__
        assert named_holder.feed_liq is liquid
        assert named_holder.seed_sol is solid
        assert liquid.name == "feed_liq"
        assert solid.name == "seed_sol"


def test_get_props_emits_deprecation_and_matches_direct_methods():
    """BasePhase.getProps warns at caller stacklevel and returns (cp, rho, enthalpy)."""
    liquid = LiquidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mass_frac=LIQUID_MASS_FRAC
    )
    for basis in ("mass", "mole"):
        with pytest.warns(DeprecationWarning, match="getProps is deprecated") as record:
            cp, rho, enthalpy = liquid.getProps(basis=basis)
        assert record[0].filename == __file__
        np.testing.assert_allclose(cp, liquid.getCp(basis=basis), rtol=PARITY_RTOL)
        np.testing.assert_allclose(rho, liquid.getDensity(basis=basis), rtol=PARITY_RTOL)
        np.testing.assert_allclose(
            enthalpy, liquid.getEnthalpy(basis=basis), rtol=PARITY_RTOL
        )


def test_get_props_phase_mix_emits_deprecation_for_liquid_and_slurry():
    """getPropsPhaseMix warns at caller stacklevel for both single-phase and slurry inputs."""
    liquid = LiquidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mass_frac=LIQUID_MASS_FRAC
    )
    solid = SolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC
    )
    expected_vfrac_solid = solid.vol / (liquid.vol + solid.vol)  # [-]
    expected_vfrac = np.array([1.0 - expected_vfrac_solid, expected_vfrac_solid])  # [-]
    expected_rho = np.array(
        [liquid.getDensity(basis="mass"), solid.getDensity(basis="mass")]
    )  # [kg/m**3]
    expected_mass = expected_vfrac * expected_rho  # [kg/m**3]
    expected_mfrac = expected_mass / expected_mass.sum()  # [-]

    for helper in (get_props_phase_mix, get_props_stream_mix):
        with pytest.warns(DeprecationWarning, match="getPropsPhaseMix") as record_single:
            cp_s, rho_s, enthalpy_s, vfrac_s, mfrac_s = helper([liquid], basis="mass")
        assert len(record_single) == 1
        assert record_single[0].filename == __file__
        np.testing.assert_allclose(
            cp_s, [liquid.getCp(basis="mass")], rtol=PARITY_RTOL
        )
        np.testing.assert_allclose(
            rho_s, [liquid.getDensity(basis="mass")], rtol=PARITY_RTOL
        )
        np.testing.assert_allclose(
            enthalpy_s, [liquid.getEnthalpy(basis="mass")], rtol=PARITY_RTOL
        )
        np.testing.assert_allclose(vfrac_s, [1.0], rtol=PARITY_RTOL)
        np.testing.assert_allclose(mfrac_s, [1.0], rtol=PARITY_RTOL)

        with pytest.warns(DeprecationWarning, match="getPropsPhaseMix") as record:
            cp, rho, enthalpy, vfrac, mfrac = helper([liquid, solid], basis="mass")
        assert len(record) == 1
        assert record[0].filename == __file__
        np.testing.assert_allclose(
            cp,
            [liquid.getCp(basis="mass"), solid.getCp(basis="mass")],
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(rho, expected_rho, rtol=PARITY_RTOL)
        np.testing.assert_allclose(
            enthalpy,
            [liquid.getEnthalpy(basis="mass"), solid.getEnthalpy(basis="mass")],
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(vfrac, expected_vfrac, rtol=PARITY_RTOL)
        np.testing.assert_allclose(mfrac, expected_mfrac, rtol=PARITY_RTOL)


def test_solid_porosity_and_stream_shims_emit_deprecation():
    """Deprecated SolidPhase/SolidStream.getPorosity and stream shims warn at caller stacklevel."""
    x_grid = np.linspace(10.0, 100.0, 10)  # [um]
    distrib = np.ones_like(x_grid) * 1e6  # [#/um]
    moments = np.array([9e7, 4.95e3, 3.3e-1, 2.475e-5])  # [m**k]

    solid_phase = SolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC
    )
    solid_phase.x_distrib = x_grid
    solid_phase.distrib = distrib
    solid_phase.moments = moments

    solid_stream = SolidStream(
        DATA_PATH, temp=SAMPLE_TEMP, mass_flow=SAMPLE_FLOW, mass_frac=SOLID_MASS_FRAC
    )
    solid_stream.x_distrib = x_grid
    solid_stream.distrib = distrib
    solid_stream.moments = moments

    with pytest.warns(
        DeprecationWarning, match="getPorosity on refactored SolidPhase"
    ) as rec_phase:
        porosity_phase = solid_phase.getPorosity(distrib=distrib)
    assert rec_phase[0].filename == __file__

    with pytest.warns(
        DeprecationWarning, match="getPorosity on refactored SolidStream"
    ) as rec_stream:
        porosity_stream = solid_stream.getPorosity(distrib=distrib)
    assert rec_stream[0].filename == __file__

    assert 0.0 < porosity_phase < 1.0
    np.testing.assert_allclose(porosity_phase, porosity_stream, rtol=PARITY_RTOL)

    t_grid = np.array([0.0, 1.0, 2.0, 3.0])  # [s]
    y_grid = np.array([10.0, 20.0, 30.0, 40.0])  # [K]
    with pytest.warns(
        DeprecationWarning, match="Interpolation in PharmaPy.Streams_Refactored"
    ) as rec_interp:
        interp_val = Interpolation(t_grid, y_grid, 1.5)
    assert rec_interp[0].filename == __file__
    np.testing.assert_allclose(interp_val, 25.0, rtol=PARITY_RTOL)

    with pytest.warns(
        DeprecationWarning, match="BatchToFlowConnector in PharmaPy.Streams_Refactored"
    ) as rec_btf:
        connector = BatchToFlowConnector(cycle_time=10.0)
    assert rec_btf[0].filename == __file__
    assert connector.cycle_time == pytest.approx(10.0)

    # Verify caller warning attribution for both LiquidStream (which overrides
    # __init__) and SolidStream (which inherits BaseStream.__init__).
    temp_offset = 5.0  # [K]
    with pytest.warns(
        DeprecationWarning, match="Passing 'controls' or 'args_control'"
    ) as rec_liq_ctrl:
        controlled_liq = LiquidStream(
            DATA_PATH,
            temp=SAMPLE_TEMP,
            mass_flow=SAMPLE_FLOW,
            mass_frac=LIQUID_MASS_FRAC,
            controls={"temp": lambda t: SAMPLE_TEMP + temp_offset},
        )
    assert rec_liq_ctrl[0].filename == __file__
    assert controlled_liq.temp == pytest.approx(SAMPLE_TEMP + temp_offset)

    with pytest.warns(
        DeprecationWarning, match="Passing 'controls' or 'args_control'"
    ) as rec_sol_ctrl:
        controlled_sol = SolidStream(
            DATA_PATH,
            temp=SAMPLE_TEMP,
            mass_flow=SAMPLE_FLOW,
            mass_frac=SOLID_MASS_FRAC,
            controls={"temp": lambda t: SAMPLE_TEMP + temp_offset},
        )
    assert rec_sol_ctrl[0].filename == __file__
    assert controlled_sol.temp == pytest.approx(SAMPLE_TEMP + temp_offset)


def test_vessel_legacy_phase_attribute_access_emits_deprecation():
    """Accessing Liquid_1, Solid_1, or Vapor_1 on MultiPhaseVessel warns at caller stacklevel."""
    liquid = LiquidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mass_frac=LIQUID_MASS_FRAC
    )
    solid = SolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC
    )
    vapor = VaporPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mole_frac=VAPOR_MOLE_FRAC
    )

    vessel = BatchReactor()
    vessel.Phases = MixedPhase([liquid, solid, vapor])

    with pytest.warns(DeprecationWarning, match="Accessing 'Liquid_1'") as rec_l:
        assert vessel.Liquid_1 is liquid
    assert rec_l[0].filename == __file__
    with pytest.warns(DeprecationWarning, match="Accessing 'Solid_1'") as rec_s:
        assert vessel.Solid_1 is solid
    assert rec_s[0].filename == __file__
    with pytest.warns(DeprecationWarning, match="Accessing 'Vapor_1'") as rec_v:
        assert vessel.Vapor_1 is vapor
    assert rec_v[0].filename == __file__


def test_moles_and_mole_flow_match_legacy_gram_to_kg_conversion(tmp_path):
    """Gate G7 (#156): moles [mol] and mole_flow [mol/s] include the 1000 g/kg factor."""
    liquid_new = LiquidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mass_frac=LIQUID_MASS_FRAC
    )
    liquid_old = LegacyLiquidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mass_frac=LIQUID_MASS_FRAC
    )
    np.testing.assert_allclose(liquid_new.moles, liquid_old.moles, rtol=PARITY_RTOL)

    solid_new = SolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC
    )
    solid_old = LegacySolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC
    )
    np.testing.assert_allclose(solid_new.moles, solid_old.moles, rtol=PARITY_RTOL)

    # Setting moles must round-trip mass in [kg].
    roundtrip = SolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, moles=solid_old.moles, mass_frac=SOLID_MASS_FRAC
    )
    np.testing.assert_allclose(roundtrip.mass, SOLID_MASS, rtol=PARITY_RTOL)

    # SolidStream on a solid-only database (no rho_liq) must support mole_flow and updatePhase.
    species_mw = 100.0  # [g/mol]
    solid_cp = 1600.0  # [J/mol/K]
    solid_rho = 1500.0  # [kg/m**3]
    initial_flow = 2.0  # [kg/s]
    updated_flow = 3.0  # [kg/s]
    solid_db = {
        "S": {"mw": species_mw, "cp_solid": [solid_cp], "rho_solid": solid_rho},
    }
    solid_db_path = tmp_path / "solid_only.json"
    solid_db_path.write_text(json.dumps(solid_db))

    stream = SolidStream(str(solid_db_path), mass_flow=initial_flow, mass_frac=[1.0])
    assert stream.mole_flow == pytest.approx(initial_flow * GRAMS_PER_KG / species_mw)
    stream.updatePhase(mass_flow=updated_flow)
    assert stream.mole_flow == pytest.approx(updated_flow * GRAMS_PER_KG / species_mw)


def test_vapor_phase_and_stream_match_legacy_enthalpy_and_density(tmp_path):
    """Gate G7 (#18 / #178): VaporPhase and VaporStream match legacy EOS density and enthalpy."""
    # Database with subcritical species A (Tc = 500 K) and supercritical species B (Tc = 250 K).
    # Each entry specifies: mw [g/mol], t_crit [K], tref_hvap [K], delta_hvap [J/mol],
    # cp_liq [J/mol/K, J/mol/K**2, J/mol/K**3, J/mol/K**4, J/mol/K**5],
    # cp_vapor [J/mol/K, J/mol/K**2, J/mol/K**3, J/mol/K**4, J/mol/K**5],
    # and p_vap Antoine coefficients [-, K, K] for ln(P [Pa]) = A - B / (T [K] + C).
    vapor_db = {
        "A": {
            "mw": 50.0,  # [g/mol]
            "t_crit": 500.0,  # [K]
            "tref_hvap": 300.0,  # [K]
            "delta_hvap": 30000.0,  # [J/mol]
            "cp_liq": [120.0, 0.1, 0.0, 0.0, 0.0],  # [J/mol/K/K**k]
            "cp_vapor": [80.0, 0.05, 0.0, 0.0, 0.0],  # [J/mol/K/K**k]
            "p_vap": [10.0, 1500.0, -50.0],  # [-, K, K]
        },
        "B": {
            "mw": 30.0,  # [g/mol]
            "t_crit": 250.0,  # [K]
            "tref_hvap": 200.0,  # [K]
            "delta_hvap": 10000.0,  # [J/mol]
            "cp_liq": [90.0, 0.05, 0.0, 0.0, 0.0],  # [J/mol/K/K**k]
            "cp_vapor": [60.0, 0.02, 0.0, 0.0, 0.0],  # [J/mol/K/K**k]
            "p_vap": [9.5, 1200.0, -40.0],  # [-, K, K]
        },
    }
    db_path = str(tmp_path / "vapor_db.json")
    with open(db_path, "w", encoding="utf-8") as handle:
        json.dump(vapor_db, handle)

    mole_frac = np.array([0.6, 0.4])  # [-]
    vapor_moles = 1.5  # [mol] (or [mol/s] for stream)
    temps = np.array([320.0, 400.0, 520.0])  # [K], straddling A's Tc = 500 K

    vapor_old = LegacyVaporPhase(
        db_path,
        temp=SAMPLE_TEMP,
        pres=SAMPLE_PRES,
        moles=vapor_moles,
        mole_frac=mole_frac,
    )
    vapor_new = VaporPhase(
        db_path,
        temp=SAMPLE_TEMP,
        pressure=SAMPLE_PRES,
        moles=vapor_moles,
        mole_frac=mole_frac,
    )
    vapor_stream = VaporStream(
        db_path,
        temp=SAMPLE_TEMP,
        pressure=SAMPLE_PRES,
        mole_flow=vapor_moles,
        mole_frac=mole_frac,
    )

    np.testing.assert_allclose(vapor_new.mass, vapor_old.mass, rtol=PARITY_RTOL)
    np.testing.assert_allclose(vapor_new.vol, vapor_old.vol, rtol=PARITY_RTOL)
    np.testing.assert_allclose(
        vapor_new.getDensity(basis="mass"),
        vapor_old.getDensity(basis="mass"),
        rtol=PARITY_RTOL,
    )
    np.testing.assert_allclose(
        vapor_new.getDensity(basis="mole"),
        vapor_old.getDensity(basis="mole"),
        rtol=PARITY_RTOL,
    )

    for basis in ("mass", "mole"):
        np.testing.assert_allclose(
            vapor_new.getHeatVaporization(temps, basis=basis),
            vapor_old.getHeatVaporization(temps, basis=basis),
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(
            vapor_new.getEnthalpy(temps, basis=basis),
            vapor_old.getEnthalpy(temps, basis=basis),
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(
            vapor_stream.getEnthalpy(temps, basis=basis),
            vapor_old.getEnthalpy(temps, basis=basis),
            rtol=PARITY_RTOL,
        )


def test_cp_and_enthalpy_analytic_overrides_none_handling_and_temp_ref(tmp_path):
    """Issues #332 and #414: analytic override checks, explicit None arguments, and temp_ref."""
    liquid = LiquidPhase(
        DATA_PATH,
        temp=SAMPLE_TEMP,
        temp_ref=CUSTOM_TEMP_REF,
        mass=SAMPLE_MASS,
        mass_frac=LIQUID_MASS_FRAC,
    )
    liquid_stream = LiquidStream(
        DATA_PATH,
        temp=SAMPLE_TEMP,
        temp_ref=CUSTOM_TEMP_REF,
        mass_flow=SAMPLE_FLOW,
        mass_frac=LIQUID_MASS_FRAC,
    )
    solid = SolidPhase(
        DATA_PATH,
        temp=SAMPLE_TEMP,
        temp_ref=CUSTOM_TEMP_REF,
        mass=SOLID_MASS,
        mass_frac=SOLID_MASS_FRAC,
    )
    solid_stream = SolidStream(
        DATA_PATH,
        temp=SAMPLE_TEMP,
        temp_ref=CUSTOM_TEMP_REF,
        mass_flow=SAMPLE_FLOW,
        mass_frac=SOLID_MASS_FRAC,
    )

    # Analytic polynomial evaluation for liquid Cp and enthalpy at SAMPLE_TEMP:
    # cp_i(T) = sum_k c_{i,k} T^k [J/mol/K], h_i(T) = sum_k c_{i,k}/(k+1) (T^{k+1} - T_ref^{k+1}) [J/mol]
    cp_coeffs = np.asarray(liquid.cp_liq, dtype=float)  # [J/mol/K/K**k]
    exponents = np.arange(cp_coeffs.shape[1])  # [-]
    int_exponents = exponents + 1  # [-]
    pure_cp_mole = cp_coeffs @ (SAMPLE_TEMP**exponents)  # [J/mol/K]
    pure_cp_mass = pure_cp_mole * GRAMS_PER_KG / liquid.mw  # [J/kg/K]
    pure_h_mole = cp_coeffs @ (
        (SAMPLE_TEMP**int_exponents - CUSTOM_TEMP_REF**int_exponents) / int_exponents
    )  # [J/mol]
    pure_h_mass = pure_h_mole * GRAMS_PER_KG / liquid.mw  # [J/kg]

    alt_mass_from_mole = (ALT_MOLE_FRAC * liquid.mw) / np.dot(
        ALT_MOLE_FRAC, liquid.mw
    )  # [-]
    alt_mole_from_mass = (ALT_MASS_FRAC / liquid.mw) / np.sum(
        ALT_MASS_FRAC / liquid.mw
    )  # [-]

    for obj in (liquid, liquid_stream):
        # mole_frac override on both bases
        np.testing.assert_allclose(
            obj.getCp(mole_frac=ALT_MOLE_FRAC, basis="mole"),
            np.dot(ALT_MOLE_FRAC, pure_cp_mole),
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(
            obj.getCp(mole_frac=ALT_MOLE_FRAC, basis="mass"),
            np.dot(alt_mass_from_mole, pure_cp_mass),
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(
            obj.getEnthalpy(mole_frac=ALT_MOLE_FRAC, basis="mole"),
            np.dot(ALT_MOLE_FRAC, pure_h_mole),
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(
            obj.getEnthalpy(mole_frac=ALT_MOLE_FRAC, basis="mass"),
            np.dot(alt_mass_from_mole, pure_h_mass),
            rtol=PARITY_RTOL,
        )
        # mass_frac override on both bases
        np.testing.assert_allclose(
            obj.getCp(mass_frac=ALT_MASS_FRAC, basis="mass"),
            np.dot(ALT_MASS_FRAC, pure_cp_mass),
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(
            obj.getCp(mass_frac=ALT_MASS_FRAC, basis="mole"),
            np.dot(alt_mole_from_mass, pure_cp_mole),
            rtol=PARITY_RTOL,
        )

    # Solid custom temp_ref contract (#332): omitted temp_ref uses self.temp_ref (305 K).
    solid_cp_coeffs = np.asarray(solid.cp_solid, dtype=float)  # [J/mol/K/K**k]
    solid_exps = np.arange(solid_cp_coeffs.shape[1]) + 1  # [-]
    expected_solid_h_mole = np.dot(
        solid.mole_frac,
        solid_cp_coeffs
        @ ((SAMPLE_TEMP**solid_exps - CUSTOM_TEMP_REF**solid_exps) / solid_exps),
    )  # [J/mol]
    expected_solid_h_298 = np.dot(
        solid.mole_frac,
        solid_cp_coeffs
        @ ((SAMPLE_TEMP**solid_exps - DEFAULT_TEMP_REF**solid_exps) / solid_exps),
    )  # [J/mol]
    for sol_obj in (solid, solid_stream):
        np.testing.assert_allclose(
            sol_obj.getEnthalpy(basis="mole"),
            expected_solid_h_mole,
            rtol=PARITY_RTOL,
        )
        np.testing.assert_allclose(
            sol_obj.getEnthalpy(temp_ref=DEFAULT_TEMP_REF, basis="mole"),
            expected_solid_h_298,
            rtol=PARITY_RTOL,
        )

    # Explicit None arguments (keyword and positional) must fall back to stored state.
    for obj in (liquid, liquid_stream, solid, solid_stream):
        for basis in ("mass", "mole"):
            expected_cp = obj.getCp(basis=basis)
            expected_h = obj.getEnthalpy(basis=basis)
            expected_rho = obj.getDensity(basis=basis)

            np.testing.assert_allclose(
                obj.getCp(mole_frac=None, basis=basis), expected_cp, rtol=PARITY_RTOL
            )
            np.testing.assert_allclose(
                obj.getCp(mass_frac=None, mole_frac=None, basis=basis),
                expected_cp,
                rtol=PARITY_RTOL,
            )
            np.testing.assert_allclose(
                obj.getCp(None, None, None, basis=basis), expected_cp, rtol=PARITY_RTOL
            )
            np.testing.assert_allclose(
                obj.getEnthalpy(temp=SAMPLE_TEMP, mole_frac=None, basis=basis),
                expected_h,
                rtol=PARITY_RTOL,
            )
            np.testing.assert_allclose(
                obj.getEnthalpy(None, None, None, None, basis=basis),
                expected_h,
                rtol=PARITY_RTOL,
            )
            np.testing.assert_allclose(
                obj.getDensity(mole_frac=None, basis=basis),
                expected_rho,
                rtol=PARITY_RTOL,
            )
            np.testing.assert_allclose(
                obj.getDensity(None, None, basis=basis),
                expected_rho,
                rtol=PARITY_RTOL,
            )

    # Zero-sentinel species (<= eps) without cp_solid data must not raise when
    # converted from a legacy SolidPhase (#414).
    db = {
        "A": {
            "mw": 100.0,  # [g/mol]
            "rho_solid": 1200.0,  # [kg/m**3]
            "cp_solid": [150.0],  # [J/mol/K]
        },
        "B": {
            "mw": 80.0,  # [g/mol]
            "rho_solid": 1100.0,  # [kg/m**3]
        },
    }
    db_path = str(tmp_path / "sentinel_solid.json")
    with open(db_path, "w", encoding="utf-8") as handle:
        json.dump(db, handle)

    legacy_solid = LegacySolidPhase(
        db_path,
        temp=SAMPLE_TEMP,
        temp_ref=CUSTOM_TEMP_REF,
        mass=SOLID_MASS,
        mass_frac=[1.0, 0.0],
    )
    refactored_solid = SolidPhase.from_legacy(legacy_solid)
    assert refactored_solid.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    np.testing.assert_allclose(
        refactored_solid.getCp(basis="mole"),
        legacy_solid.getCp(basis="mole"),
        rtol=PARITY_RTOL,
    )
    np.testing.assert_allclose(
        refactored_solid.getEnthalpy(basis="mole"),
        legacy_solid.getEnthalpy(basis="mole"),
        rtol=PARITY_RTOL,
    )


def test_legacy_conversion_preserves_pressure_and_temp_ref():
    """to_legacy and from_legacy preserve non-default pressure [Pa] and temp_ref [K]."""
    liquid = LiquidPhase(
        DATA_PATH,
        temp=CUSTOM_TEMP_REF,
        pressure=ELEVATED_PRES,
        temp_ref=CUSTOM_TEMP_REF,
        mass=SAMPLE_MASS,
        mass_frac=LIQUID_MASS_FRAC,
    )
    cp_coeffs = np.asarray(liquid.cp_liq, dtype=float)  # [J/mol/K/K**k]
    int_exps = np.arange(cp_coeffs.shape[1]) + 1  # [-]
    expected_liq_h_298 = np.dot(
        liquid.mass_frac,
        (
            cp_coeffs
            @ ((CUSTOM_TEMP_REF**int_exps - DEFAULT_TEMP_REF**int_exps) / int_exps)
        )
        * GRAMS_PER_KG
        / liquid.mw,
    )  # [J/kg]
    assert liquid.getEnthalpy(basis="mass") == pytest.approx(0.0, abs=1e-12)
    legacy_liquid = liquid.to_legacy()
    assert legacy_liquid.pres == pytest.approx(ELEVATED_PRES)
    assert legacy_liquid.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    assert legacy_liquid.getEnthalpy(basis="mass") == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        legacy_liquid.getEnthalpy(temp_ref=DEFAULT_TEMP_REF, basis="mass"),
        expected_liq_h_298,
        rtol=PARITY_RTOL,
    )
    roundtrip_liquid = LiquidPhase.from_legacy(legacy_liquid)
    assert roundtrip_liquid.pres == pytest.approx(ELEVATED_PRES)
    assert roundtrip_liquid.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    assert roundtrip_liquid.getEnthalpy(basis="mass") == pytest.approx(0.0, abs=1e-12)

    liquid_stream = LiquidStream(
        DATA_PATH,
        temp=CUSTOM_TEMP_REF,
        pressure=ELEVATED_PRES,
        temp_ref=CUSTOM_TEMP_REF,
        mass_flow=SAMPLE_FLOW,
        mass_frac=LIQUID_MASS_FRAC,
    )
    legacy_liquid_stream = liquid_stream.to_legacy()
    assert legacy_liquid_stream.pres == pytest.approx(ELEVATED_PRES)
    assert legacy_liquid_stream.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    assert legacy_liquid_stream.getEnthalpy(basis="mass") == pytest.approx(
        0.0, abs=1e-12
    )
    np.testing.assert_allclose(
        legacy_liquid_stream.getEnthalpy(temp_ref=DEFAULT_TEMP_REF, basis="mass"),
        expected_liq_h_298,
        rtol=PARITY_RTOL,
    )
    from_legacy_liq_stream = LiquidPhase.from_legacy(legacy_liquid_stream)
    assert from_legacy_liq_stream.pres == pytest.approx(ELEVATED_PRES)
    assert from_legacy_liq_stream.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    assert from_legacy_liq_stream.getEnthalpy(basis="mass") == pytest.approx(
        0.0, abs=1e-12
    )

    vapor = VaporPhase(
        DATA_PATH,
        temp=CUSTOM_TEMP_REF,
        pressure=ELEVATED_PRES,
        temp_ref=CUSTOM_TEMP_REF,
        mass=SAMPLE_MASS,
        mole_frac=VAPOR_MOLE_FRAC,
    )
    # At temp == temp_ref == CUSTOM_TEMP_REF (subcritical for all present species),
    # sensible heat is 0 and vapor enthalpy equals pure Watson latent heat at CUSTOM_TEMP_REF.
    used_idx = np.flatnonzero(VAPOR_MOLE_FRAC > 0)
    watson_factor = (
        (vapor.t_crit[used_idx] - CUSTOM_TEMP_REF)
        / (vapor.t_crit[used_idx] - vapor.tref_hvap[used_idx])
    ) ** 0.38  # [-]
    latent_mass = (
        watson_factor
        * vapor.delta_hvap[used_idx]
        * GRAMS_PER_KG
        / vapor.mw[used_idx]
    )  # [J/kg]
    expected_vap_h_custom = np.dot(vapor.mass_frac[used_idx], latent_mass)  # [J/kg]
    sensible_mass_298 = (
        (
            cp_coeffs[used_idx]
            @ ((CUSTOM_TEMP_REF**int_exps - DEFAULT_TEMP_REF**int_exps) / int_exps)
        )
        * GRAMS_PER_KG
        / vapor.mw[used_idx]
    )  # [J/kg]
    expected_vap_h_298 = np.dot(
        vapor.mass_frac[used_idx], sensible_mass_298 + latent_mass
    )  # [J/kg]

    legacy_vapor = vapor.to_legacy()
    assert legacy_vapor.pres == pytest.approx(ELEVATED_PRES)
    assert legacy_vapor.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    np.testing.assert_allclose(legacy_vapor.vol, vapor.vol, rtol=PARITY_RTOL)
    np.testing.assert_allclose(
        legacy_vapor.getDensity(basis="mass"),
        vapor.getDensity(basis="mass"),
        rtol=PARITY_RTOL,
    )
    np.testing.assert_allclose(
        legacy_vapor.getEnthalpy(basis="mass"),
        expected_vap_h_custom,
        rtol=PARITY_RTOL,
    )
    np.testing.assert_allclose(
        legacy_vapor.getEnthalpy(temp_ref=DEFAULT_TEMP_REF, basis="mass"),
        expected_vap_h_298,
        rtol=PARITY_RTOL,
    )

    roundtrip_vapor = VaporPhase.from_legacy(legacy_vapor)
    assert roundtrip_vapor.pres == pytest.approx(ELEVATED_PRES)
    assert roundtrip_vapor.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    np.testing.assert_allclose(roundtrip_vapor.vol, vapor.vol, rtol=PARITY_RTOL)
    np.testing.assert_allclose(
        roundtrip_vapor.getEnthalpy(basis="mass"),
        expected_vap_h_custom,
        rtol=PARITY_RTOL,
    )

    vapor_stream = VaporStream(
        DATA_PATH,
        temp=CUSTOM_TEMP_REF,
        pressure=ELEVATED_PRES,
        temp_ref=CUSTOM_TEMP_REF,
        mass_flow=SAMPLE_FLOW,
        mole_frac=VAPOR_MOLE_FRAC,
    )
    legacy_vapor_stream = vapor_stream.to_legacy()
    assert legacy_vapor_stream.pres == pytest.approx(ELEVATED_PRES)
    assert legacy_vapor_stream.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    np.testing.assert_allclose(
        legacy_vapor_stream.vol_flow, vapor_stream.vol_flow, rtol=PARITY_RTOL
    )
    np.testing.assert_allclose(
        legacy_vapor_stream.getEnthalpy(basis="mass"),
        expected_vap_h_custom,
        rtol=PARITY_RTOL,
    )
    np.testing.assert_allclose(
        legacy_vapor_stream.getEnthalpy(temp_ref=DEFAULT_TEMP_REF, basis="mass"),
        expected_vap_h_298,
        rtol=PARITY_RTOL,
    )
    from_legacy_vap_stream = VaporPhase.from_legacy(legacy_vapor_stream)
    assert from_legacy_vap_stream.pres == pytest.approx(ELEVATED_PRES)
    assert from_legacy_vap_stream.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    np.testing.assert_allclose(
        from_legacy_vap_stream.getEnthalpy(basis="mass"),
        expected_vap_h_custom,
        rtol=PARITY_RTOL,
    )

    solid = SolidPhase(
        DATA_PATH,
        temp=CUSTOM_TEMP_REF,
        pressure=ELEVATED_PRES,
        temp_ref=CUSTOM_TEMP_REF,
        mass=SOLID_MASS,
        mass_frac=SOLID_MASS_FRAC,
    )
    solid_cp_coeffs = np.asarray(solid.cp_solid, dtype=float)  # [J/mol/K/K**k]
    solid_exps = np.arange(solid_cp_coeffs.shape[1]) + 1  # [-]
    expected_solid_h_298 = np.dot(
        solid.mass_frac,
        (
            solid_cp_coeffs
            @ ((CUSTOM_TEMP_REF**solid_exps - DEFAULT_TEMP_REF**solid_exps) / solid_exps)
        )
        * GRAMS_PER_KG
        / solid.mw,
    )  # [J/kg]
    assert solid.getEnthalpy(basis="mass") == pytest.approx(0.0, abs=1e-12)
    legacy_solid = solid.to_legacy()
    assert legacy_solid.pres == pytest.approx(ELEVATED_PRES)
    assert legacy_solid.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    assert legacy_solid.getEnthalpy(basis="mass") == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        legacy_solid.getEnthalpy(temp_ref=DEFAULT_TEMP_REF, basis="mass"),
        expected_solid_h_298,
        rtol=PARITY_RTOL,
    )

    roundtrip_solid = SolidPhase.from_legacy(legacy_solid)
    assert roundtrip_solid.pres == pytest.approx(ELEVATED_PRES)
    assert roundtrip_solid.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    assert roundtrip_solid.getEnthalpy(basis="mass") == pytest.approx(0.0, abs=1e-12)

    solid_stream = SolidStream(
        DATA_PATH,
        temp=CUSTOM_TEMP_REF,
        pressure=ELEVATED_PRES,
        temp_ref=CUSTOM_TEMP_REF,
        mass_flow=SAMPLE_FLOW,
        mass_frac=SOLID_MASS_FRAC,
    )
    legacy_solid_stream = solid_stream.to_legacy()
    assert legacy_solid_stream.pres == pytest.approx(ELEVATED_PRES)
    assert legacy_solid_stream.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    assert legacy_solid_stream.getEnthalpy(basis="mass") == pytest.approx(
        0.0, abs=1e-12
    )
    np.testing.assert_allclose(
        legacy_solid_stream.getEnthalpy(temp_ref=DEFAULT_TEMP_REF, basis="mass"),
        expected_solid_h_298,
        rtol=PARITY_RTOL,
    )
    from_legacy_sol_stream = SolidPhase.from_legacy(legacy_solid_stream)
    assert from_legacy_sol_stream.pres == pytest.approx(ELEVATED_PRES)
    assert from_legacy_sol_stream.temp_ref == pytest.approx(CUSTOM_TEMP_REF)
    assert from_legacy_sol_stream.getEnthalpy(basis="mass") == pytest.approx(
        0.0, abs=1e-12
    )


def test_to_stream_to_phase_roundtrip_and_nested_mixed_stream():
    """All three phase families round-trip through to_stream()/to_phase() and nested MixedStream."""
    liquid = LiquidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SAMPLE_MASS, mass_frac=LIQUID_MASS_FRAC
    )
    solid = SolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC
    )
    vapor_mass = 0.85  # [kg]
    vapor = VaporPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=vapor_mass, mole_frac=VAPOR_MOLE_FRAC
    )

    for phase, stream_cls, phase_cls in (
        (liquid, LiquidStream, LiquidPhase),
        (solid, SolidStream, SolidPhase),
        (vapor, VaporStream, VaporPhase),
    ):
        as_stream = phase.to_stream()
        assert isinstance(as_stream, stream_cls)
        assert as_stream.is_stream
        back_to_phase = as_stream.to_phase()
        assert isinstance(back_to_phase, phase_cls)
        assert not back_to_phase.is_stream
        np.testing.assert_allclose(back_to_phase.mass, phase.mass, rtol=PARITY_RTOL)

    # Nested MixedStream flattening with unequal flow rates and ordering check.
    flow_1 = 0.15  # [kg/s]
    flow_2 = 0.35  # [kg/s]
    flow_3 = 0.50  # [kg/s]
    stream_1 = LiquidStream(
        DATA_PATH, temp=SAMPLE_TEMP, mass_flow=flow_1, mass_frac=LIQUID_MASS_FRAC
    )
    stream_2 = LiquidStream(
        DATA_PATH, temp=SAMPLE_TEMP, mass_flow=flow_2, mass_frac=ALT_MASS_FRAC
    )
    stream_3 = SolidStream(
        DATA_PATH, temp=SAMPLE_TEMP, mass_flow=flow_3, mass_frac=SOLID_MASS_FRAC
    )

    nested = MixedStream([MixedStream([stream_1, stream_2]), stream_3])
    assert list(nested.Streams) == [stream_1, stream_2, stream_3]
    assert [s.mass_flow for s in nested.Streams] == pytest.approx(
        [flow_1, flow_2, flow_3]
    )
    np.testing.assert_allclose(
        nested.mass_flow, flow_1 + flow_2 + flow_3, rtol=PARITY_RTOL
    )
    expected_mix_frac = (
        flow_1 * LIQUID_MASS_FRAC + flow_2 * ALT_MASS_FRAC + flow_3 * SOLID_MASS_FRAC
    ) / (flow_1 + flow_2 + flow_3)  # [-]
    np.testing.assert_allclose(nested.mass_frac, expected_mix_frac, rtol=PARITY_RTOL)

    re_phased = nested.to_phase()
    assert isinstance(re_phased, MixedPhase)
    assert [type(p) for p in re_phased.Phases] == [
        LiquidPhase,
        LiquidPhase,
        SolidPhase,
    ]
    np.testing.assert_allclose(
        re_phased.mass, flow_1 + flow_2 + flow_3, rtol=PARITY_RTOL
    )


def test_reactor_rxn_kinetics_validates_liquid_phase_presence():
    """Setting RxnKinetics without a liquid phase raises RuntimeError naming intraphase_processes."""
    reactor = BatchReactor()
    reactor.Phases = SolidPhase(
        DATA_PATH, temp=SAMPLE_TEMP, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC
    )
    rate_constant = 1e-2  # [L/mol/s]
    activation_energy = 0.0  # [J/mol]
    kinetics = RxnKinetics(
        path=DATA_PATH,
        rxn_list=["A + B --> C"],
        k_params=rate_constant,
        ea_params=activation_energy,
    )
    with pytest.raises(RuntimeError, match="intraphase_processes"):
        reactor.RxnKinetics = kinetics
