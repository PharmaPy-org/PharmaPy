"""Regression tests for the crystal shape factor in mixed-phase volumes.

Issue #52: ``Slurry.Phases`` and ``Cake.Phases`` convert the third distribution
moment ``mu_3`` into a physical solid volume without multiplying by the
volumetric shape factor ``kv``. The physical relation is

    V_solid = kv * mu_3,

so both quantities are wrong by a factor ``kv`` for any non-spherical-cube
habit (``kv != 1``). ``Slurry.getFractions`` and ``Slurry.getSolidsConcentr``
already apply ``kv``, so the defect also makes ``Slurry`` internally
inconsistent.

Fixtures use the shared ``tests/Flowsheet/data/compound_database.json`` thermo
file and real ``LiquidPhase``/``SolidPhase`` collaborators; only ``kv`` and the
moments/distribution are chosen to make the missing factor visible.
"""

import numpy as np
import pytest

from PharmaPy.MixedPhases import Cake, Slurry
from PharmaPy.Phases import LiquidPhase, SolidPhase


pytestmark = pytest.mark.unit


# Volumetric shape factor of the crystals [-]. Chosen well away from the
# default 1.0 so that an omitted kv changes the result by a factor of two.
KV_TEST = 0.5

# Liquid composition [-]: the database defines species A, B, C, D, solvent.
LIQUID_MASS_FRAC = [0.1, 0.1, 0.1, 0.1, 0.6]

# Solid composition [-]: pure component A.
SOLID_MASS_FRAC = [1.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def thermo_path(data_path):
    """Path of the shared pure-component thermodynamic database."""
    return str(data_path["flowsheet"] / "compound_database.json")


def test_slurry_liquid_volume_uses_shape_factor(thermo_path):
    """Slurry liquid volume must subtract kv * mu_3, not mu_3.

    The slurry moments are volume-specific, so ``mu_3`` [m**3/m**3] becomes a
    solid volume fraction only after multiplication by ``kv``.
    """
    vol_slurry = 1.0e-3  # [m**3]

    # Volume-specific moments [1/m**3, m/m**3, m**2/m**3, m**3/m**3].
    mu_three = 0.2  # [m**3/m**3]
    moments = np.array([1.0e12, 1.0e6, 1.0e3, mu_three])

    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_MASS_FRAC)
    solid = SolidPhase(thermo_path, mass_frac=SOLID_MASS_FRAC,
                       moments=moments, kv=KV_TEST)

    slurry = Slurry(vol=vol_slurry, moments=moments)
    slurry.Phases = (liquid, solid)

    # Hand-computed: solid volume fraction = kv * mu_3 = 0.5 * 0.2 = 0.1 [-],
    # so the liquid occupies 0.9 * 1.0e-3 m**3.
    expected_vol_liq = 9.0e-4  # [m**3]

    assert slurry.Liquid_1.vol == pytest.approx(expected_vol_liq, rel=1e-12)

    # Internal consistency: getFractions already applies kv, so the liquid
    # volume assigned by the Phases setter must agree with it.
    vol_frac_liq = slurry.getFractions()[0]  # [-]
    assert slurry.Liquid_1.vol == pytest.approx(vol_slurry * vol_frac_liq,
                                                rel=1e-12)


def test_cake_volume_uses_shape_factor(thermo_path):
    """Cake volume must be kv * mu_3 / (1 - porosity), not mu_3 / (1 - porosity).

    The cake solid fraction is ``1 - porosity`` by definition, so the solid
    volume recovered from the reported cake volume must equal ``kv * mu_3``.
    """
    # Crystal size grid [um] and a number-based CSD [#/um] (mass=0 keeps the
    # distribution unconverted, so the moments follow directly from it).
    x_distrib = np.linspace(10.0, 500.0, 40)  # [um]
    mean_size = 200.0  # [um]
    std_size = 60.0  # [um]
    # Peak number density [#/um], scaled so the cake volume is of the order of
    # one litre.
    peak_density = 8.0e5
    distrib = peak_density * np.exp(
        -0.5 * ((x_distrib - mean_size) / std_size)**2)

    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_MASS_FRAC)
    solid = SolidPhase(thermo_path, mass_frac=SOLID_MASS_FRAC,
                       x_distrib=x_distrib, distrib=distrib, kv=KV_TEST)

    cake = Cake()
    cake.Phases = (liquid, solid)

    # Independently integrated from the fixed Gaussian CSD above:
    # trapz(distrib * x**3, x) * (1e-6)**3.
    expected_mu_three = 1.2224284763295205e-3  # [m**3]
    assert solid.moments[3] == pytest.approx(expected_mu_three, rel=1e-12)

    mu_three = solid.moments[3]  # [m**3]
    expected_cake_vol = KV_TEST * mu_three / (1 - cake.porosity)  # [m**3]

    assert cake.cake_vol == pytest.approx(expected_cake_vol, rel=1e-12)

    # Equivalent statement of the same invariant, free of the porosity model:
    # the solid occupies a fraction (1 - porosity) of the cake.
    vol_solid = KV_TEST * mu_three  # [m**3]
    assert cake.cake_vol * (1 - cake.porosity) == pytest.approx(vol_solid,
                                                               rel=1e-12)
