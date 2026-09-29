import json

import numpy as np
import pytest

from PharmaPy.Phases import VaporPhase


THERMO_TWO_SPECIES = {
    "light": {
        "mw": 18.0,
        "t_crit": 650.0,
        "rho_liq": 1000.0,
        "cp_liq": [75.0],
        "p_vap": [8.0, 1500.0, -40.0],
        "delta_hvap": 40000.0,
        "tref_hvap": 350.0,
    },
    "heavy": {
        "mw": 100.0,
        "t_crit": 700.0,
        "rho_liq": 900.0,
        "cp_liq": [150.0],
        "p_vap": [8.0, 1800.0, -40.0],
        "delta_hvap": 60000.0,
        "tref_hvap": 350.0,
    },
}


def test_vaporphase_mass_basis_weights_latent_heat_by_mass_fraction(tmp_path):
    path = tmp_path / "thermo_two_species.json"
    path.write_text(json.dumps(THERMO_TWO_SPECIES))

    temp = 350.0
    temp_ref = 298.15
    mole_frac = np.array([0.5, 0.5])
    phase = VaporPhase(
        str(path),
        temp=temp,
        moles=1.0,
        mole_frac=mole_frac,
        check_input=False,
    )

    mw = np.array([18.0, 100.0])
    cp_liq = np.array([75.0, 150.0])
    delta_hvap = np.array([40000.0, 60000.0])
    mass_frac = mole_frac * mw / np.dot(mole_frac, mw)

    species_h_mole = cp_liq * (temp - temp_ref) + delta_hvap
    expected_h_mole = np.dot(mole_frac, species_h_mole)

    species_h_mass = species_h_mole * 1000.0 / mw
    expected_h_mass = np.dot(mass_frac, species_h_mass)

    h_mole = phase.getEnthalpy(temp=temp, basis="mole")
    h_mass = phase.getEnthalpy(temp=temp, basis="mass")

    assert h_mole == pytest.approx(expected_h_mole)
    assert h_mass == pytest.approx(expected_h_mass)
    assert h_mass == pytest.approx(h_mole * 1000.0 / phase.mw_av)

    np.testing.assert_allclose(
        phase.getEnthalpy(temp=temp, total_h=False, basis="mole"),
        np.atleast_2d(species_h_mole),
    )
    np.testing.assert_allclose(
        phase.getEnthalpy(temp=temp, total_h=False, basis="mass"),
        np.atleast_2d(species_h_mass),
    )
