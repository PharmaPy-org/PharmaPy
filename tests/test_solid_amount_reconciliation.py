"""Robustness tests for solid amount reconciliation across public APIs.

Solid mass, volume, and moles, together with their stream flow aliases, must
describe the same inventory after explicit-amount, distribution, and slurry
phase updates. Fixtures use the shared
``tests/Flowsheet/data/compound_database.json`` thermo file and real phase and
stream collaborators. Expected amounts are derived independently from the
mass-basis mixing rules and the exact kg-to-g conversion.
"""

import numpy as np
import pytest

from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream


pytestmark = pytest.mark.unit

# Solid composition [-] over the database species (A, B, C, D, solvent).
# Deliberately a two-component mixture with unequal molecular weights and
# unequal pure densities, so an average property cannot coincide with a pure
# component value.
SOLID_MASS_FRAC = [0.8, 0.2, 0.0, 0.0, 0.0]

# Liquid composition [-]; only needs to be a valid non-degenerate mixture.
LIQUID_MASS_FRAC = [0.1, 0.1, 0.1, 0.1, 0.6]

# Pure-component values read from tests/Flowsheet/data/compound_database.json.
MW_A = 100.0  # [g/mol]
MW_B = 50.0  # [g/mol]
RHO_SOLID_A = 1230.0  # [kg/m**3]
RHO_SOLID_B = 864.7  # [kg/m**3]
LIQUID_DENSITIES = np.array([
    1230.0,
    864.7,
    1200.0,
    867.1,
    887.6,
])  # [kg/m**3], ordered as A, B, C, D, solvent

# Mass-basis mixing rules, evaluated by hand for SOLID_MASS_FRAC:
#   1 / mw_av   = sum(w_i / mw_i)   = 0.8/100 + 0.2/50    [mol/g]
#   1 / rho_av  = sum(w_i / rho_i)  = 0.8/1230 + 0.2/864.7 [m**3/kg]
INV_MW_AV = (SOLID_MASS_FRAC[0] / MW_A
             + SOLID_MASS_FRAC[1] / MW_B)  # [mol/g] = 0.012
INV_RHO_SOLID = (SOLID_MASS_FRAC[0] / RHO_SOLID_A
                 + SOLID_MASS_FRAC[1] / RHO_SOLID_B)  # [m**3/kg]
INV_RHO_LIQUID = np.dot(
    LIQUID_MASS_FRAC,
    1 / LIQUID_DENSITIES,
)  # [m**3/kg]


@pytest.fixture(scope="module")
def thermo_path(data_path):
    """Path of the shared pure-component thermodynamic database."""
    return str(data_path["flowsheet"] / "compound_database.json")


def test_solid_phase_update_mass_refreshes_moles(thermo_path):
    """``updatePhase(mass=...)`` must keep ``moles`` on the new mass.

    ``LiquidPhase.updatePhase`` reconciles mass, volume, and moles through
    ``__set_amounts``; the solid phase must offer the same guarantee.
    """
    mass_initial = 2.0  # [kg]
    mass_updated = 5.0  # [kg]

    phase = SolidPhase(thermo_path, mass=mass_initial,
                       mass_frac=SOLID_MASS_FRAC)

    # Constructor behavior from #50 / PR #155, restated here as the baseline
    # this update path must preserve: 2 kg * 1000 g/kg * 0.012 mol/g.
    assert phase.moles == pytest.approx(24.0, rel=1e-12)  # [mol]

    phase.updatePhase(mass=mass_updated)

    assert phase.mass == pytest.approx(mass_updated, rel=1e-12)  # [kg]

    expected_vol = mass_updated * INV_RHO_SOLID  # [m**3]
    assert phase.vol == pytest.approx(expected_vol, rel=1e-12)

    # 5 kg * 1000 g/kg * 0.012 mol/g = 60 mol.
    expected_moles = mass_updated * 1000 * INV_MW_AV  # [mol]
    assert expected_moles == pytest.approx(60.0, rel=1e-12)
    assert phase.moles == pytest.approx(expected_moles, rel=1e-12)


def test_solid_phase_update_distribution_refreshes_moles(thermo_path):
    """A distribution update that changes ``mass`` must also change ``moles``.

    ``updatePhase(distrib=...)`` back-calculates volume from the third moment
    and mass from the density, so the mole amount has to follow that new mass
    even though no explicit ``mass`` argument was given.
    """
    kv = 1.0  # [-], volumetric shape factor
    x_distrib = np.linspace(10.0, 500.0, 40)  # [um]
    distrib = np.full_like(x_distrib, 1.0e6)  # [#/um], flat number density

    phase = SolidPhase(thermo_path, mass=0, mass_frac=SOLID_MASS_FRAC,
                       x_distrib=x_distrib, distrib=distrib, kv=kv)

    # Doubling a number-based distribution doubles every moment, hence the
    # solid volume and mass, hence the mole amount.
    mass_before = phase.mass  # [kg]
    moles_before = phase.moles  # [mol]
    assert mass_before > 0
    assert moles_before == pytest.approx(mass_before * 1000 * INV_MW_AV,
                                         rel=1e-12)

    phase.updatePhase(distrib=2 * distrib)

    assert phase.mass == pytest.approx(2 * mass_before, rel=1e-12)  # [kg]

    expected_moles = 2 * moles_before  # [mol]
    assert phase.moles == pytest.approx(expected_moles, rel=1e-12)


def test_slurry_stream_phases_refresh_solid_mole_flow(thermo_path):
    """``SlurryStream.Phases`` must keep ``mole_flow`` on the assigned mass flow.

    The setter distributes the stream volumetric flow between the phases and
    derives a solid mass flow; the paired mole flow must be reconciled by the
    setter rather than left to the caller.
    """
    vol_flow = 1.0e-3  # [m**3/s]
    kv = 1.0  # [-]
    mu_three = 0.2  # [m**3/m**3], volume-specific third moment
    # Volume-specific moments [1/m**3, m/m**3, m**2/m**3, m**3/m**3]; only the
    # third order enters the phase-volume split.
    moments = np.array([1.0e12, 1.0e6, 1.0e3, mu_three])

    liquid = LiquidStream(thermo_path, mass_frac=LIQUID_MASS_FRAC)
    solid = SolidStream(thermo_path, mass_frac=SOLID_MASS_FRAC, kv=kv)

    stream = SlurryStream(vol_flow=vol_flow, moments=moments)
    stream.Phases = (liquid, solid)

    # Solid volume fraction is kv * mu_3 = 0.2 [-], so 2.0e-4 m**3/s of solid.
    expected_vol_flow_solid = kv * mu_three * vol_flow  # [m**3/s]
    expected_mass_flow = expected_vol_flow_solid / INV_RHO_SOLID  # [kg/s]

    assert solid.mass_flow == pytest.approx(expected_mass_flow, rel=1e-10)

    expected_mole_flow = expected_mass_flow * 1000 * INV_MW_AV  # [mol/s]
    assert solid.mole_flow == pytest.approx(expected_mole_flow, rel=1e-10)


def test_slurry_phases_refresh_solid_inventory(thermo_path):
    """A moment-based batch slurry must reconcile solid amount bases."""
    slurry_vol = 1.0e-3  # [m**3]
    kv = 1.0  # [-]
    mu_three = 0.2  # [m**3/m**3], volume-specific third moment
    # Volume-specific moments [1/m**3, m/m**3, m**2/m**3, m**3/m**3].
    moments = np.array([1.0e12, 1.0e6, 1.0e3, mu_three])

    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_MASS_FRAC)
    solid = SolidPhase(thermo_path, mass_frac=SOLID_MASS_FRAC, kv=kv)

    slurry = Slurry(vol=slurry_vol, moments=moments)
    slurry.Phases = (liquid, solid)

    expected_solid_vol = kv * mu_three * slurry_vol  # [m**3]
    expected_solid_mass = expected_solid_vol / INV_RHO_SOLID  # [kg]
    expected_solid_moles = (expected_solid_mass * 1000
                            * INV_MW_AV)  # [mol]

    assert solid.vol == pytest.approx(expected_solid_vol, rel=1e-10)
    assert solid.mass == pytest.approx(expected_solid_mass, rel=1e-10)
    assert solid.moles == pytest.approx(expected_solid_moles, rel=1e-10)


@pytest.mark.parametrize("amount_keyword", ("mass", "mass_flow"))
def test_solid_stream_update_mass_refreshes_flow_aliases(
        thermo_path, amount_keyword):
    """Either mass-flow keyword must refresh every existing flow alias."""
    mass_flow_initial = 2.0  # [kg/s]
    mass_flow_updated = 5.0  # [kg/s]

    stream = SolidStream(
        thermo_path,
        mass_flow=mass_flow_initial,
        mass_frac=SOLID_MASS_FRAC,
    )
    stream.vol_flow = 0.0  # [m**3/s], deliberately stale alias

    stream.updatePhase(**{amount_keyword: mass_flow_updated})

    expected_mole_flow = mass_flow_updated * 1000 * INV_MW_AV  # [mol/s]
    expected_vol_flow = mass_flow_updated * INV_RHO_SOLID  # [m**3/s]
    assert stream.mass_flow == pytest.approx(mass_flow_updated, rel=1e-12)
    assert stream.mole_flow == pytest.approx(expected_mole_flow, rel=1e-12)
    assert stream.vol_flow == pytest.approx(expected_vol_flow, rel=1e-12)


def test_solid_stream_update_rejects_conflicting_mass_keywords(thermo_path):
    """The additive mass-flow alias must reject an ambiguous amount update."""
    mass_flow_initial = 2.0  # [kg/s]
    inherited_mass_flow = 3.0  # [kg/s]
    aliased_mass_flow = 4.0  # [kg/s]
    stream = SolidStream(
        thermo_path,
        mass_flow=mass_flow_initial,
        mass_frac=SOLID_MASS_FRAC,
    )

    with pytest.raises(ValueError, match="either 'mass' or 'mass_flow'"):
        stream.updatePhase(
            mass=inherited_mass_flow,
            mass_flow=aliased_mass_flow,
        )


def test_slurry_stream_distribution_refreshes_solid_mole_flow(thermo_path):
    """The distribution-based slurry path must reconcile solid flow bases."""
    vol_flow = 1.0e-3  # [m**3/s]
    kv = 0.5  # [-], non-unit volumetric shape factor
    x_distrib = np.array([0.0, 100.0, 200.0, 300.0])  # [um]
    # Volume-specific number distribution [#/m**3/um]. Its third moment is
    # 0.2 m**3/m**3 by the trapezoidal rule on the 100 um grid:
    # 100 * (1e9 * 100**3 + 1.25e8 * 200**3) * 1e-18 = 0.2.
    distrib = np.array([0.0, 1.0e9, 1.25e8, 0.0])  # [#/m**3/um]
    third_moment = 0.2  # [m**3/m**3]

    liquid = LiquidStream(thermo_path, mass_frac=LIQUID_MASS_FRAC)
    solid = SolidStream(thermo_path, mass_frac=SOLID_MASS_FRAC, kv=kv)

    stream = SlurryStream(
        vol_flow=vol_flow,
        x_distrib=x_distrib,
        distrib=distrib,
    )
    stream.Phases = (liquid, solid)

    expected_vol_flow_solid = kv * third_moment * vol_flow  # [m**3/s]
    expected_mass_flow = expected_vol_flow_solid / INV_RHO_SOLID  # [kg/s]
    expected_mole_flow = expected_mass_flow * 1000 * INV_MW_AV  # [mol/s]

    assert solid.mass_flow == pytest.approx(expected_mass_flow, rel=1e-10)
    assert solid.mole_flow == pytest.approx(expected_mole_flow, rel=1e-10)


def test_slurry_stream_distribution_accepts_mass_slurry_basis(thermo_path):
    """Reconcile a distribution when only total slurry mass flow is known.

    Parameters
    ----------
    thermo_path : str
        Path to the shared pure-component thermodynamic database.
    """
    total_mass_flow = 1.0  # [kg/s]
    kv = 0.5  # [-], non-unit volumetric shape factor
    x_distrib = np.array([0.0, 100.0, 200.0, 300.0])  # [um]
    # Volume-specific number distribution [#/m**3/um] with third moment
    # 0.2 m**3/m**3 by the trapezoidal rule on the 100 um grid.
    distrib = np.array([0.0, 1.0e9, 1.25e8, 0.0])  # [#/m**3/um]
    third_moment = 0.2  # [m**3/m**3]

    liquid = LiquidStream(thermo_path, mass_frac=LIQUID_MASS_FRAC)
    solid = SolidStream(thermo_path, mass_frac=SOLID_MASS_FRAC, kv=kv)
    stream = SlurryStream(x_distrib=x_distrib, distrib=distrib)
    stream.mass_slurry = total_mass_flow  # [kg/s]

    stream.Phases = (liquid, solid)

    solid_volume_fraction = kv * third_moment  # [-]
    liquid_density = 1 / INV_RHO_LIQUID  # [kg/m**3]
    solid_density = 1 / INV_RHO_SOLID  # [kg/m**3]
    slurry_density = (
        (1 - solid_volume_fraction) * liquid_density
        + solid_volume_fraction * solid_density
    )  # [kg/m**3]
    expected_vol_flow = total_mass_flow / slurry_density  # [m**3/s]
    expected_solid_vol_flow = (
        solid_volume_fraction * expected_vol_flow
    )  # [m**3/s]
    expected_solid_mass_flow = (
        expected_solid_vol_flow * solid_density
    )  # [kg/s]
    expected_solid_mole_flow = (
        expected_solid_mass_flow * 1000 * INV_MW_AV
    )  # [mol/s]

    assert stream.vol == pytest.approx(expected_vol_flow, rel=1e-10)
    assert stream.mass_flow == pytest.approx(total_mass_flow, rel=1e-12)
    assert solid.vol_flow == pytest.approx(expected_solid_vol_flow, rel=1e-10)
    assert solid.mass_flow == pytest.approx(expected_solid_mass_flow,
                                            rel=1e-10)
    assert solid.mole_flow == pytest.approx(expected_solid_mole_flow,
                                            rel=1e-10)
