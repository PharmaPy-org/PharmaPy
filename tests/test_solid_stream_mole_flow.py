"""Regression test for solid-stream mass-to-mole flow conversion."""

import json

import pytest

from PharmaPy.Streams import SolidStream


_MOLECULAR_WEIGHT = 100.0  # [g/mol]
_SOLID_HEAT_CAPACITY = 1600.0  # [J/mol/K], constant polynomial coefficient
_SOLID_DENSITY = 1500.0  # [kg/m**3]
_PURE_MASS_FRACTION = [1.0]  # [-]


@pytest.fixture
def path_thermo(tmp_path):
    """Write a minimal single-component solid property database."""
    database = {
        'solid': {
            'mw': _MOLECULAR_WEIGHT,
            'cp_solid': [_SOLID_HEAT_CAPACITY],
            'rho_solid': _SOLID_DENSITY,
        },
    }
    path = tmp_path / 'solid_stream_db.json'
    path.write_text(json.dumps(database))

    return str(path)


def test_solid_stream_mole_flow_uses_corrected_phase_moles(path_thermo):
    """SolidStream must expose the phase conversion on a flow basis."""
    mass_flow = 2.0  # [kg/s]

    stream = SolidStream(
        path_thermo,
        mass_flow=mass_flow,
        mass_frac=_PURE_MASS_FRACTION,
    )

    expected_mole_flow = 20.0  # [mol/s] = 2 kg/s * 1000 g/kg / 100 g/mol
    assert stream.mole_flow == pytest.approx(expected_mole_flow)
