"""Regression tests for ``SolidPhase`` mole accounting.

``SolidPhase.moles`` must be derived from the phase mass that the constructor
finally settles on (explicit ``mass`` argument, or the mass back-calculated
from the third moment when ``mass=0``), and it must apply the same kg-to-g
conversion that ``LiquidPhase``/``VaporPhase`` use, since molecular weights are
stored in g/mol while phase masses are in kg.
"""

import json

import pytest

from PharmaPy.Phases import SolidPhase


# Two solids with identical density so the density mixing rule cannot change
# the moments-derived mass, but distinct molecular weights so that the average
# molecular weight is a genuine mixture property.
_MW_A = 100.0  # [g/mol]
_MW_B = 50.0  # [g/mol]
_RHO_SOLID = 1500.0  # [kg/m**3]

_MASS_FRAC = [0.8, 0.2]  # [-]

# Mass-basis average molecular weight, computed independently of PharmaPy as
# the harmonic-style mixing rule mw_av = 1 / sum(w_i / mw_i).
_INV_MW_AV = _MASS_FRAC[0] / _MW_A + _MASS_FRAC[1] / _MW_B  # [mol/g]


@pytest.fixture
def path_thermo(tmp_path):
    """Write a minimal two-component solid thermophysical database."""
    database = {
        'A': {'mw': _MW_A, 'cp_solid': [1600.0], 'rho_solid': _RHO_SOLID},
        'B': {'mw': _MW_B, 'cp_solid': [1600.0], 'rho_solid': _RHO_SOLID},
    }

    path = tmp_path / 'solid_db.json'
    path.write_text(json.dumps(database))

    return str(path)


def test_solid_phase_moles_from_explicit_mass(path_thermo):
    """An explicitly given mass must be converted kg -> g before dividing."""
    mass = 2.0  # [kg]

    phase = SolidPhase(path_thermo, mass=mass, mass_frac=_MASS_FRAC)

    expected_mw_av = 1 / _INV_MW_AV  # [g/mol]
    assert phase.mw_av == pytest.approx(expected_mw_av)

    expected_moles = mass * 1000 * _INV_MW_AV  # [mol] = 24.0 mol
    assert phase.moles == pytest.approx(expected_moles)


def test_solid_phase_moles_from_moment_derived_mass(path_thermo):
    """With ``mass=0`` the mass comes from the moments; moles must follow it."""
    kv = 1.0  # [-]
    mom_three = 2e-3  # [m**3 crystal/m**3], third moment of the CSD
    moments = [1.0, 1.0, 1.0, mom_three]  # [m**k/m**3] for k = 0..3

    phase = SolidPhase(path_thermo, mass=0, mass_frac=_MASS_FRAC,
                       moments=moments, kv=kv)

    expected_mass = mom_three * kv * _RHO_SOLID  # [kg] = 3.0 kg
    assert phase.mass == pytest.approx(expected_mass)

    expected_moles = expected_mass * 1000 * _INV_MW_AV  # [mol] = 36.0 mol
    assert phase.moles == pytest.approx(expected_moles)
