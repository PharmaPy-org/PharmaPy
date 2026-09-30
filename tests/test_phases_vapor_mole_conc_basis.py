"""Vapor-phase constructor composition and state-retention contracts.

The tests cover molar-concentration conversion, equivalent fraction inputs,
stream forwarding, and constructor pressure used by downstream property
calculations. The self-contained two-species fixture needs no repository data.
"""

import json

import numpy as np
import pytest

from PharmaPy.Phases import VaporPhase
from PharmaPy.Streams import VaporStream
from conftest import THERMO_TWO_SPECIES


pytestmark = pytest.mark.unit


def test_vapor_phase_mole_conc_uses_molar_basis(tmp_path):
    path = tmp_path / "thermo_two_species.json"
    path.write_text(json.dumps(THERMO_TWO_SPECIES))

    mw = np.array([18.0, 100.0])  # [g/mol]
    mole_conc = np.array([2.0, 3.0])  # [mol/L]
    moles = 1.0  # [mol]

    # With no solvent index, molar concentrations normalize directly to mole
    # fractions; mass fractions follow from the molar-weighted mixture mass.
    expected_mole_frac = mole_conc / mole_conc.sum()  # [-]
    expected_mass_frac = (
        expected_mole_frac * mw / np.dot(expected_mole_frac, mw))  # [-]
    expected_mw_av = np.dot(expected_mole_frac, mw)  # [g/mol]
    expected_mass = moles * expected_mw_av / 1000  # [kg]

    phase = VaporPhase(str(path), moles=moles, mole_conc=mole_conc)

    expected_concentration = expected_mole_frac * phase.pres / (8.314 * phase.temp) / 1000  # [mol/L], ideal gas, R [J/mol/K]
    np.testing.assert_allclose(phase.mole_conc, expected_concentration)
    np.testing.assert_allclose(phase.mole_frac, expected_mole_frac)
    np.testing.assert_allclose(phase.mass_frac, expected_mass_frac)
    assert phase.mw_av == pytest.approx(expected_mw_av)
    assert phase.mass == pytest.approx(expected_mass)

    # VaporStream forwards mole_conc to VaporPhase, so it must agree.
    stream = VaporStream(str(path), mole_flow=moles, mole_conc=mole_conc)

    np.testing.assert_allclose(stream.mole_frac, expected_mole_frac)
    assert stream.mass_flow == pytest.approx(expected_mass)


def test_vapor_phase_mole_conc_matches_equivalent_fractions(tmp_path):
    path = tmp_path / "thermo_two_species.json"
    path.write_text(json.dumps(THERMO_TWO_SPECIES))

    mw = np.array([18.0, 100.0])  # [g/mol]
    mole_conc = np.array([2.0, 3.0])  # [mol/L]
    equivalent_mole_frac = mole_conc / mole_conc.sum()  # [-]
    equivalent_mass_frac = (
        equivalent_mole_frac
        * mw
        / np.dot(equivalent_mole_frac, mw)
    )  # [-]
    moles = 1.0  # [mol]

    phase_from_conc = VaporPhase(
        str(path), moles=moles, mole_conc=mole_conc
    )
    phase_from_frac = VaporPhase(
        str(path), moles=moles, mole_frac=equivalent_mole_frac
    )
    phase_from_mass_frac = VaporPhase(
        str(path), moles=moles, mass_frac=equivalent_mass_frac
    )

    np.testing.assert_allclose(
        phase_from_conc.mole_frac, phase_from_frac.mole_frac
    )
    np.testing.assert_allclose(
        phase_from_conc.mass_frac, phase_from_frac.mass_frac
    )
    assert phase_from_conc.mw_av == pytest.approx(phase_from_frac.mw_av)
    np.testing.assert_allclose(
        phase_from_conc.mole_frac, phase_from_mass_frac.mole_frac
    )
    np.testing.assert_allclose(
        phase_from_conc.mass_frac, phase_from_mass_frac.mass_frac
    )
    assert phase_from_conc.mw_av == pytest.approx(
        phase_from_mass_frac.mw_av
    )


def test_vapor_phase_retains_pressure_for_default_dew_point(tmp_path):
    path = tmp_path / "thermo_two_species.json"
    path.write_text(json.dumps(THERMO_TWO_SPECIES))

    # This pressure keeps both synthetic saturation temperatures subcritical.
    pressure = 1.0e5  # [Pa]
    mole_conc = np.array([2.0, 3.0])  # [mol/L]
    moles = 1.0  # [mol]
    phase = VaporPhase(
        str(path), pres=pressure, moles=moles, mole_conc=mole_conc
    )

    dew_point_explicit = phase.getDewPoint(pres=pressure)  # [K]

    assert phase.pres == pytest.approx(pressure)
    assert phase.getDewPoint() == pytest.approx(dew_point_explicit)
