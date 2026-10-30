"""Tests for species-keyed phase compositions, retrieve_field, and solvent blends (#336, #338).

Covers:
- Species-keyed composition mappings (`mass_frac`, `mole_frac`, `mass_conc`,
  `mole_conc`) on `LiquidPhase`, `SolidPhase`, `VaporPhase`, `LiquidStream`,
  `SolidStream`, and `VaporStream` (`__init__` and `updatePhase`).
- Solute concentration (`mole_conc` or `mass_conc`) combined with a
  solute-free solvent-blend `mass_frac` mapping (#336), including the analytical
  benchmark from #336 and the single-solvent `rawlings_9-5.ipynb` pattern.
- `.retrieve_field(field_name, species=None, copy=True, as_dict=False)` across
  phases and streams, including profile axes, copy isolation, and dual
  `(KeyError, ValueError)` lookup errors (#338).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.Streams import LiquidStream, SolidStream, VaporStream


REACTOR_DB = str(
    Path(__file__).resolve().parent / 'integration' / 'data' / ' compounds_reactor.json'
).replace(' compounds_reactor.json', 'compounds_reactor.json')
PURE_DB = str(
    Path(__file__).resolve().parent.parent
    / 'examples'
    / 'data'
    / 'Playgrounds'
    / 'Pure_components_properties.json'
)


@pytest.fixture()
def solvent_blend_db(tmp_path: Path) -> str:
    """Create the four-component database from issue #336 comment 6200391817."""
    t_crit = 650.0  # [K], critical temperature for constant-property fixture
    p_crit = 4.0e6  # [Pa], critical pressure
    data = {
        'A': {
            'mw': 150.0,  # [g/mol]
            't_crit': t_crit,
            'p_crit': p_crit,
            'rho_liq': 1200.0,  # [kg/m**3]
            'rho_solid': 1350.0,  # [kg/m**3]
            'cp_liq': [200.0, 0.0, 0.0, 0.0, 0.0],  # [J/mol/K]
            'cp_solid': [180.0, 0.0, 0.0],  # [J/kg/K]
            'cp_vapor': [150.0, 0.0, 0.0, 0.0, 0.0],  # [J/mol/K]
            'p_sat': [10.0, 3000.0, -50.0],
            'delta_hvap': 40000.0,  # [J/mol]
            't_nbp': 450.0,  # [K]
        },
        'B': {
            'mw': 200.0,  # [g/mol]
            't_crit': t_crit,
            'p_crit': p_crit,
            'rho_liq': 1100.0,  # [kg/m**3]
            'rho_solid': 1250.0,  # [kg/m**3]
            'cp_liq': [220.0, 0.0, 0.0, 0.0, 0.0],  # [J/mol/K]
            'cp_solid': [190.0, 0.0, 0.0],  # [J/kg/K]
            'cp_vapor': [160.0, 0.0, 0.0, 0.0, 0.0],  # [J/mol/K]
            'p_sat': [10.0, 3100.0, -50.0],
            'delta_hvap': 42000.0,  # [J/mol]
            't_nbp': 460.0,  # [K]
        },
        'ACN': {
            'mw': 41.05,  # [g/mol]
            't_crit': t_crit,
            'p_crit': p_crit,
            'rho_liq': 786.0,  # [kg/m**3]
            'rho_solid': 900.0,  # [kg/m**3]
            'cp_liq': [91.0, 0.0, 0.0, 0.0, 0.0],  # [J/mol/K]
            'cp_solid': [140.0, 0.0, 0.0],  # [J/kg/K]
            'cp_vapor': [70.0, 0.0, 0.0, 0.0, 0.0],  # [J/mol/K]
            'p_sat': [10.0, 2900.0, -45.0],
            'delta_hvap': 33000.0,  # [J/mol]
            't_nbp': 354.8,  # [K]
        },
        'THF': {
            'mw': 72.11,  # [g/mol]
            't_crit': t_crit,
            'p_crit': p_crit,
            'rho_liq': 889.0,  # [kg/m**3]
            'rho_solid': 980.0,  # [kg/m**3]
            'cp_liq': [124.0, 0.0, 0.0, 0.0, 0.0],  # [J/mol/K]
            'cp_solid': [150.0, 0.0, 0.0],  # [J/kg/K]
            'cp_vapor': [95.0, 0.0, 0.0, 0.0, 0.0],  # [J/mol/K]
            'p_sat': [10.0, 2850.0, -45.0],
            'delta_hvap': 32000.0,  # [J/mol]
            't_nbp': 339.0,  # [K]
        },
    }
    db_file = tmp_path / 'solvent_blend_336.json'
    db_file.write_text(json.dumps(data), encoding='utf-8')
    return str(db_file)


@pytest.mark.unit
def test_solvent_blend_analytical_benchmark_issue_336(solvent_blend_db: str) -> None:
    """Verify the analytical 4-component solvent-blend example from #336."""
    conc_a = 1.0  # [mol/L]
    conc_b = 1.0  # [mol/L]
    w_acn_blend = 0.3  # [-], solute-free solvent blend mass fraction
    w_thf_blend = 0.7  # [-], solute-free solvent blend mass fraction
    vol_holdup = 0.1  # [m**3]

    phase = LiquidPhase(
        path_thermo=solvent_blend_db,
        mole_conc={'B': conc_b, 'A': conc_a},
        mass_frac={'THF': w_thf_blend, 'ACN': w_acn_blend},
        vol=vol_holdup,
    )

    # Independent analytical calculation (issue #336 comment 6200391817)
    mw_a = 150.0  # [g/mol]
    mw_b = 200.0  # [g/mol]
    mw_acn = 41.05  # [g/mol]
    mw_thf = 72.11  # [g/mol]
    rho_a = 1200.0  # [kg/m**3]
    rho_b = 1100.0  # [kg/m**3]
    rho_acn = 786.0  # [kg/m**3]
    rho_thf = 889.0  # [kg/m**3]

    c_mass_a = conc_a * mw_a  # [kg/m**3]
    c_mass_b = conc_b * mw_b  # [kg/m**3]
    phi_solutes = c_mass_a / rho_a + c_mass_b / rho_b  # [-]
    phi_solvents = 1.0 - phi_solutes  # [-]
    rho_solv_mix = 1.0 / (w_acn_blend / rho_acn + w_thf_blend / rho_thf)  # [kg/m**3]

    c_mass_acn = w_acn_blend * rho_solv_mix * phi_solvents  # [kg/m**3]
    c_mass_thf = w_thf_blend * rho_solv_mix * phi_solvents  # [kg/m**3]
    expected_mass_conc = np.array([c_mass_a, c_mass_b, c_mass_acn, c_mass_thf])  # [kg/m**3]

    rho_mix = float(np.sum(expected_mass_conc))  # [kg/m**3]
    expected_mass_frac = expected_mass_conc / rho_mix  # [-]
    expected_mole_conc = expected_mass_conc / np.array([mw_a, mw_b, mw_acn, mw_thf])  # [mol/L]
    expected_mole_frac = expected_mole_conc / np.sum(expected_mole_conc)  # [-]

    assert phase.mass_conc == pytest.approx(expected_mass_conc, rel=1e-12)
    assert phase.mass_frac == pytest.approx(expected_mass_frac, rel=1e-12)
    assert phase.mole_conc == pytest.approx(expected_mole_conc, rel=1e-12)
    assert phase.mole_frac == pytest.approx(expected_mole_frac, rel=1e-12)
    assert phase.getDensity() == pytest.approx(rho_mix, rel=1e-12)
    assert phase.mass == pytest.approx(vol_holdup * rho_mix, rel=1e-12)


@pytest.mark.unit
def test_single_solvent_mapping_matches_name_solv(solvent_blend_db: str) -> None:
    """Single-solvent `mass_frac={'THF': 1.0}` matches legacy `name_solv='THF'`."""
    vol_holdup = 0.05  # [m**3]
    phase_dict = LiquidPhase(
        path_thermo=solvent_blend_db,
        mole_conc={'A': 0.8, 'B': 0.4},
        mass_frac={'THF': 1.0},
        vol=vol_holdup,
    )
    phase_legacy = LiquidPhase(
        path_thermo=solvent_blend_db,
        mole_conc=[0.8, 0.4, 0.0, 0.0],
        name_solv='THF',
        vol=vol_holdup,
    )

    assert phase_dict.mass_frac == pytest.approx(phase_legacy.mass_frac, rel=1e-12)
    assert phase_dict.mole_frac == pytest.approx(phase_legacy.mole_frac, rel=1e-12)
    assert phase_dict.mole_conc == pytest.approx(phase_legacy.mole_conc, rel=1e-12)
    assert phase_dict.mass_conc == pytest.approx(phase_legacy.mass_conc, rel=1e-12)
    assert phase_dict.mass == pytest.approx(phase_legacy.mass, rel=1e-12)
    assert phase_dict.moles == pytest.approx(phase_legacy.moles, rel=1e-12)


@pytest.mark.unit
def test_solvent_blend_mass_conc_and_update_phase(solvent_blend_db: str) -> None:
    """Solvent-blend conversion also works with `mass_conc` and in `updatePhase`."""
    vol_holdup = 0.1  # [m**3]
    phase = LiquidPhase(
        path_thermo=solvent_blend_db,
        mass_conc={'A': 150.0, 'B': 200.0},
        mass_frac={'ACN': 0.3, 'THF': 0.7},
        vol=vol_holdup,
    )
    assert phase.mole_conc[0] == pytest.approx(1.0, rel=1e-12)
    assert phase.mole_conc[1] == pytest.approx(1.0, rel=1e-12)

    # Update phase with a new solvent blend and solute concentration
    phase.updatePhase(
        mole_conc={'A': 0.5},
        mass_frac={'ACN': 0.5, 'THF': 0.5},
        vol=vol_holdup,
    )
    assert phase.mole_conc[0] == pytest.approx(0.5, rel=1e-12)
    assert phase.mole_conc[1] == pytest.approx(0.0, abs=1e-15)
    # Verify solvent mass ratio in the resulting mixture is 1:1
    assert phase.mass_frac[2] == pytest.approx(phase.mass_frac[3], rel=1e-12)


@pytest.mark.unit
def test_solvent_blend_validation_errors(solvent_blend_db: str) -> None:
    """Reject invalid solvent blends, overlapping solute/solvent, or infeasible conc."""
    vol_holdup = 0.1  # [m**3]

    # Solvent mass fractions do not sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        LiquidPhase(
            path_thermo=solvent_blend_db,
            mole_conc={'A': 0.5},
            mass_frac={'ACN': 0.3, 'THF': 0.5},
            vol=vol_holdup,
        )

    # Overlapping solute and solvent species
    with pytest.raises(ValueError, match="both solute concentrations and solvent mass_frac"):
        LiquidPhase(
            path_thermo=solvent_blend_db,
            mole_conc={'A': 0.5, 'THF': 0.2},
            mass_frac={'THF': 1.0},
            vol=vol_holdup,
        )

    # Physically infeasible solute concentration (volume fraction >= 1.0)
    with pytest.raises(ValueError, match="volume fraction"):
        LiquidPhase(
            path_thermo=solvent_blend_db,
            mole_conc={'A': 9.0},  # 9 mol/L * 150 g/mol = 1350 kg/m^3 > rho_A (1200)
            mass_frac={'THF': 1.0},
            vol=vol_holdup,
        )

    # Both name_solv and solvent-blend mass_frac provided
    with pytest.raises(ValueError, match="not both"):
        LiquidPhase(
            path_thermo=solvent_blend_db,
            mole_conc={'A': 0.5},
            mass_frac={'THF': 1.0},
            name_solv='THF',
            vol=vol_holdup,
        )

    # Positional array mass_frac alongside mole_conc still raises RuntimeWarning
    with pytest.raises(RuntimeWarning, match="More than one measure of composition"):
        LiquidPhase(
            path_thermo=solvent_blend_db,
            mole_conc={'A': 0.5},
            mass_frac=[0.0, 0.0, 0.3, 0.7],
            vol=vol_holdup,
        )


@pytest.mark.unit
def test_species_keyed_compositions_across_phases_and_streams(
    solvent_blend_db: str,
) -> None:
    """Verify species-keyed composition mappings across Liquid, Solid, Vapor, and Streams."""
    # 1. LiquidPhase with mole_frac dict (sparse + reordered)
    liq = LiquidPhase(
        path_thermo=solvent_blend_db,
        mole_frac={'THF': 0.75, 'A': 0.25},
        moles=2.0,
    )
    assert liq.mole_frac == pytest.approx([0.25, 0.0, 0.0, 0.75])

    # 2. SolidPhase with mass_frac dict and mole_frac dict
    sol = SolidPhase(
        path_thermo=solvent_blend_db,
        mass_frac={'B': 0.6, 'A': 0.4},
        mass=1.5,
        moments=[1e6, 1e2, 1e-2, 1e-6],
    )
    assert sol.mass_frac[0] == pytest.approx(0.4)
    assert sol.mass_frac[1] == pytest.approx(0.6)
    sol.updatePhase(mass_frac={'A': 1.0})
    assert sol.mass_frac[0] == pytest.approx(1.0)

    sol_mole = SolidPhase(
        path_thermo=solvent_blend_db,
        mole_frac={'A': 0.5, 'B': 0.5},
        mass=1.0,
        moments=[1e6, 1e2, 1e-2, 1e-6],
    )
    assert sol_mole.mole_frac[:2] == pytest.approx([0.5, 0.5])

    # 3. VaporPhase with mole_frac and mass_frac dicts
    vap = VaporPhase(
        path_thermo=solvent_blend_db,
        mole_frac={'ACN': 0.4, 'THF': 0.6},
        moles=1.0,
    )
    assert vap.mole_frac == pytest.approx([0.0, 0.0, 0.4, 0.6])
    vap.updatePhase(mass_frac={'ACN': 0.5, 'THF': 0.5})
    assert vap.mass_frac == pytest.approx([0.0, 0.0, 0.5, 0.5])

    # 4. Streams (LiquidStream, SolidStream, VaporStream)
    liq_stream = LiquidStream(
        path_thermo=solvent_blend_db,
        mole_conc={'A': 0.5, 'B': 0.25},
        mass_frac={'ACN': 0.4, 'THF': 0.6},
        vol_flow=0.01,
    )
    assert liq_stream.mole_conc[:2] == pytest.approx([0.5, 0.25])
    liq_stream.updatePhase(mole_conc={'A': 0.2}, mass_frac={'THF': 1.0})
    assert liq_stream.mole_conc[0] == pytest.approx(0.2)

    sol_stream = SolidStream(
        path_thermo=solvent_blend_db,
        mass_frac={'A': 0.7, 'B': 0.3},
        mass_flow=0.5,
    )
    assert sol_stream.mass_frac[:2] == pytest.approx([0.7, 0.3])
    sol_stream.updatePhase(mass_frac={'B': 1.0})
    assert sol_stream.mass_frac[1] == pytest.approx(1.0)

    vap_stream = VaporStream(
        path_thermo=solvent_blend_db,
        mole_frac={'ACN': 0.25, 'THF': 0.75},
        mole_flow=0.5,
    )
    assert vap_stream.mole_frac == pytest.approx([0.0, 0.0, 0.25, 0.75])
    vap_stream.updatePhase(mole_frac={'ACN': 0.8, 'THF': 0.2})
    assert vap_stream.mole_frac == pytest.approx([0.0, 0.0, 0.8, 0.2])


@pytest.mark.unit
def test_composition_mapping_validation_and_profiles(solvent_blend_db: str) -> None:
    """Validate profile mappings and error messages for invalid species mappings."""
    # Spatial profile mapping (3 axial points)
    liq_prof = LiquidPhase(
        path_thermo=solvent_blend_db,
        mole_conc={'A': [0.2, 0.4, 0.6], 'B': [0.1, 0.1, 0.1]},
        mass_frac={'THF': 1.0},
        vol=0.1,
    )
    assert liq_prof.mole_conc.shape == (3, 4)
    assert liq_prof.mole_conc[:, 0] == pytest.approx([0.2, 0.4, 0.6])
    assert liq_prof.mole_conc[:, 1] == pytest.approx([0.1, 0.1, 0.1])

    # Unknown species key raises ValueError naming valid species
    with pytest.raises(ValueError, match="Unknown species.*'UNKNOWN'"):
        LiquidPhase(
            path_thermo=solvent_blend_db,
            mass_frac={'UNKNOWN': 1.0},
            mass=1.0,
        )

    # Empty composition dict raises ValueError
    with pytest.raises(ValueError, match="mapping must not be empty"):
        LiquidPhase(
            path_thermo=solvent_blend_db,
            mass_frac={},
            mass=1.0,
        )

    # Negative value in composition dict raises ValueError
    with pytest.raises(ValueError, match="non-negative"):
        LiquidPhase(
            path_thermo=solvent_blend_db,
            mass_frac={'A': -0.1, 'THF': 1.1},
            mass=1.0,
        )


@pytest.mark.unit
def test_retrieve_field_across_phases_and_profiles(solvent_blend_db: str) -> None:
    """Verify `.retrieve_field()` for scalar, species-indexed, and profile states (#338)."""
    liq = LiquidPhase(
        path_thermo=solvent_blend_db,
        mole_conc={'A': [0.2, 0.5], 'B': [0.1, 0.3]},
        mass_frac={'ACN': 0.3, 'THF': 0.7},
        vol=0.1,
        temp=310.0,
    )

    # Full array retrieval and copy isolation
    mc = liq.retrieve_field('mole_conc')
    assert mc.shape == (2, 4)
    mc[0, 0] = 999.0
    assert liq.mole_conc[0, 0] == pytest.approx(0.2)

    # Single-species retrieval
    mc_a = liq.retrieve_field('mole_conc', species='A')
    assert mc_a == pytest.approx([0.2, 0.5])

    # Reordered subset retrieval as array and as dict
    mc_ba = liq.retrieve_field('mole_conc', species=['B', 'A'])
    assert mc_ba.shape == (2, 2)
    assert mc_ba[:, 0] == pytest.approx([0.1, 0.3])
    assert mc_ba[:, 1] == pytest.approx([0.2, 0.5])

    mc_dict = liq.retrieve_field('mole_conc', species=['B', 'A'], as_dict=True)
    assert list(mc_dict.keys()) == ['B', 'A']
    assert mc_dict['B'] == pytest.approx([0.1, 0.3])
    assert mc_dict['A'] == pytest.approx([0.2, 0.5])

    # Scalar field retrieval
    assert liq.retrieve_field('temp') == pytest.approx(310.0)
    assert liq.retrieve_field('vol') == pytest.approx(0.1)

    # Scalar field rejects species filter and as_dict=True
    with pytest.raises(ValueError, match="not indexed by species"):
        liq.retrieve_field('temp', species='A')
    with pytest.raises(ValueError, match="not indexed by species"):
        liq.retrieve_field('temp', as_dict=True)

    # Unknown field raises both KeyError and ValueError
    with pytest.raises(KeyError, match="Field 'nonexistent' does not exist"):
        liq.retrieve_field('nonexistent')
    with pytest.raises(ValueError, match="Field 'nonexistent' does not exist"):
        liq.retrieve_field('nonexistent')

    # Unknown species raises both KeyError and ValueError
    with pytest.raises(KeyError, match="Unknown species 'NO_SUCH'"):
        liq.retrieve_field('mole_conc', species='NO_SUCH')
    with pytest.raises(ValueError, match="Unknown species 'NO_SUCH'"):
        liq.retrieve_field('mole_conc', species='NO_SUCH')

    # copy=False with as_dict=True on a species subset returns views into stored array
    dict_views = liq.retrieve_field(
        'mole_conc', species=['B', 'A'], copy=False, as_dict=True
    )
    assert np.shares_memory(dict_views['B'], liq.mole_conc)
    assert np.shares_memory(dict_views['A'], liq.mole_conc)


@pytest.mark.unit
def test_solid_composition_update_reconciles_volume_and_stream_vol_flow(
    solvent_blend_db: str,
) -> None:
    """Composition-only updates on SolidPhase and SolidStream reconcile volume with density."""
    # Database species A has rho_solid=1350 kg/m**3 (MW=150 g/mol);
    # B has rho_solid=1250 kg/m**3 (MW=200 g/mol).
    x_grid = np.linspace(10.0, 50.0, 5)  # [um]
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # [-]
    solid_mass = 1.5  # [kg]

    solid = SolidPhase(
        path_thermo=solvent_blend_db,
        mass=solid_mass,
        x_distrib=x_grid,
        distrib=weights,
        mass_frac={'A': 1.0},
    )
    assert solid.vol == pytest.approx(solid_mass / 1350.0)
    assert solid.moles == pytest.approx(solid_mass * 1000.0 / 150.0)

    # Update to pure B via mass_frac mapping
    solid.updatePhase(mass_frac={'B': 1.0})
    assert solid.mass == pytest.approx(solid_mass)
    assert solid.getDensity() == pytest.approx(1250.0)
    assert solid.vol == pytest.approx(solid_mass / 1250.0)
    assert solid.moles == pytest.approx(solid_mass * 1000.0 / 200.0)

    # Update back to pure A via mole_frac mapping
    solid.updatePhase(mole_frac={'A': 1.0})
    assert solid.vol == pytest.approx(solid_mass / 1350.0)
    assert solid.moles == pytest.approx(solid_mass * 1000.0 / 150.0)

    # SolidStream volume-flow alias stays synchronized on composition-only update
    stream = SolidStream(
        path_thermo=solvent_blend_db,
        mass_flow=solid_mass,
        x_distrib=x_grid,
        distrib=weights,
        mass_frac={'A': 1.0},
    )
    assert stream.vol_flow == pytest.approx(solid_mass / 1350.0)
    stream.updatePhase(mass_frac={'B': 1.0})
    assert stream.mass_flow == pytest.approx(solid_mass)
    assert stream.vol_flow == pytest.approx(solid_mass / 1250.0)
    assert stream.mole_flow == pytest.approx(solid_mass * 1000.0 / 200.0)


@pytest.mark.unit
def test_profile_update_preserves_solvent_blend_and_vapor_mass_conc_init(
    solvent_blend_db: str,
) -> None:
    """2-D profile updates on LiquidPhase/LiquidStream retain stored solvent blend."""
    liq = LiquidPhase(
        path_thermo=solvent_blend_db,
        mole_conc={'A': [0.2, 0.4], 'B': [0.1, 0.2]},
        mass_frac={'ACN': 0.3, 'THF': 0.7},
        vol=0.1,
    )
    initial_mole_conc = liq.mole_conc.copy()  # [mol/L]
    initial_mass = np.array(liq.mass, copy=True)  # [kg]
    assert liq.concentration_basis is False
    assert np.all(initial_mole_conc[:, 2:] > 0.0)

    # Repeat 2-D profile update via mole_conc mapping without re-passing mass_frac
    liq.updatePhase(mole_conc={'A': [0.2, 0.4], 'B': [0.1, 0.2]})
    assert liq.concentration_basis is False
    np.testing.assert_allclose(liq.mole_conc, initial_mole_conc, rtol=1e-12)
    np.testing.assert_allclose(liq.mass, initial_mass, rtol=1e-12)

    # Repeat 2-D profile update via mass_conc mapping (MW_A=150 g/mol, MW_B=200 g/mol)
    target_mass_conc = {
        'A': initial_mole_conc[:, 0] * 150.0,
        'B': initial_mole_conc[:, 1] * 200.0,
    }  # [kg/m**3]
    liq.updatePhase(mass_conc=target_mass_conc)
    assert liq.concentration_basis is False
    np.testing.assert_allclose(liq.mole_conc, initial_mole_conc, rtol=1e-12)
    np.testing.assert_allclose(liq.mass, initial_mass, rtol=1e-12)

    # VaporPhase and VaporStream accept mass_conc mappings in __init__
    vap = VaporPhase(
        path_thermo=solvent_blend_db,
        mass_conc={'A': 2.0, 'B': 1.0},
        vol=1.0,
    )
    assert vap.mass_frac == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0])
    vap_stream = VaporStream(
        path_thermo=solvent_blend_db,
        mass_conc={'A': 2.0, 'B': 1.0},
        vol_flow=1.0,
    )
    assert vap_stream.mass_frac == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0])
