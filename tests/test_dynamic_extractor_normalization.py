"""Regression tests for DynamicExtractor mole-fraction closure."""

import json

import numpy as np
import pytest

import PharmaPy.DynamicExtraction as dynamic_extraction
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


_SPECIES = {
    "a": {
        "mw": 18.0,  # [g/mol]
        "rho_liq": 1000.0,  # [kg/m**3]
        "t_crit": 647.0,  # [K]
        "cp_liq": [75.0],  # [J/mol/K]
        "p_vap": [8.0, 1500.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 40000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
    "b": {
        "mw": 150.0,  # [g/mol]
        "rho_liq": 1200.0,  # [kg/m**3]
        "t_crit": 800.0,  # [K]
        "cp_liq": [220.0],  # [J/mol/K]
        "p_vap": [8.0, 2200.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 70000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
    "c": {
        "mw": 60.0,  # [g/mol]
        "rho_liq": 800.0,  # [kg/m**3]
        "t_crit": 700.0,  # [K]
        "cp_liq": [130.0],  # [J/mol/K]
        "p_vap": [8.0, 1800.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 55000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
}
# Synthetic species are chosen only to provide distinct molar volumes and heat
# capacities for real `LiquidPhase` construction; they are not a calibrated
# liquid-liquid extraction system.


def _write_thermo_database(tmp_path):
    """Write the synthetic liquid-property database for real collaborator tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory that receives the generated JSON database.

    Returns
    -------
    str
        Absolute path to the thermodynamic-property database [-].
    """
    path = tmp_path / "dynamic_extractor_thermo.json"
    path.write_text(json.dumps(_SPECIES))

    return str(path)


def _constant_distribution(x_light, x_heavy, temp):
    """Return asymmetric distribution coefficients for the test fixture.

    Parameters
    ----------
    x_light, x_heavy : ndarray
        Light- and heavy-phase mole fractions [-].
    temp : ndarray
        Stage temperatures [K].

    Returns
    -------
    ndarray
        Component distribution coefficients [-].
    """
    coeffs = np.array([0.4, 1.5, 1.1])  # [-]

    return np.broadcast_to(coeffs, np.shape(x_light))  # [-]


def test_material_balances_replace_dependent_components_with_closure():
    """The dependent x and y residuals enforce mole-fraction sums."""
    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=1,
        k_fun=_constant_distribution,
        eff=1.0,
    )

    # Deliberately unnormalized so a constant-zero mutant cannot satisfy the
    # dependent light-phase closure assertion.
    x_i = np.array([[0.2, 0.3, 0.55]])  # [-]
    m_i = _constant_distribution(x_i, x_i, np.array([298.15]))  # [-]
    y_i = m_i * x_i  # [-]

    x_in = np.array([0.30, 0.30, 0.40])  # [-]
    y_in = np.array([0.20, 0.20, 0.60])  # [-]
    light_flows = np.array([10.0, 10.0])  # [mol/s]
    heavy_flows = np.array([8.0, 8.0])  # [mol/s]
    holdup_light = np.array([12.0])  # [mol]
    holdup_heavy = np.array([6.0])  # [mol]

    x_augm = np.vstack((x_in, x_i))  # [-]
    y_augm = np.vstack((y_i, y_in))  # [-]
    temp_augm = np.array([295.0, 298.15, 301.0])  # [K]
    augm_arrays = (x_augm, y_augm, temp_augm, light_flows, heavy_flows)

    residuals = extractor.material_balances(
        0.0,
        x_i=x_i,
        y_i=y_i,
        holdup_light=holdup_light,
        holdup_heavy=holdup_heavy,
        u_int=np.array([0.0]),
        temp=np.array([298.15]),
        di_sdot=None,
        augm_arrays=augm_arrays,
    )

    x_residuals, y_residuals = residuals
    expected_independent_dxdt = np.array([
        [0.1361111111111111, -0.09523809523809523],
    ])  # [1/s]
    # Values are hand-calculated from [1.96, -2.0] mol/s component imbalances
    # over [14.4, 21.0] mol effective holdups.

    np.testing.assert_allclose(x_residuals[:, :-1],
                               expected_independent_dxdt)
    np.testing.assert_allclose(x_residuals[:, -1], x_i.sum(axis=1) - 1.0)
    np.testing.assert_allclose(y_residuals[:, :-1],
                               m_i[:, :-1] * x_i[:, :-1] - y_i[:, :-1])
    np.testing.assert_allclose(y_residuals[:, -1], y_i.sum(axis=1) - 1.0)


def test_initialize_model_returns_closed_phase_compositions(tmp_path):
    """The initialization correction closes the real batch-extractor result."""
    thermo_path = _write_thermo_database(tmp_path)
    feed_mole_frac = np.array([0.30, 0.30, 0.40])  # [-]
    solvent_mole_frac = np.array([0.20, 0.20, 0.60])  # [-]
    feed_reference_moles = 10.0  # [mol]
    solvent_reference_moles = 8.0  # [mol]
    batch_moles = feed_reference_moles + solvent_reference_moles  # [mol]
    batch_mole_frac = (
        feed_reference_moles * feed_mole_frac
        + solvent_reference_moles * solvent_mole_frac
    ) / batch_moles  # [-]
    feed_flow = 10.0  # [mol/s]
    solvent_flow = 8.0  # [mol/s]

    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=1,
        k_fun=_constant_distribution,
        eff=1.0,
    )
    phase = LiquidPhase(
        thermo_path,
        mole_frac=batch_mole_frac,
        moles=batch_moles,
        temp=298.15,  # [K]
        verbose=False,
    )
    feed = LiquidStream(
        thermo_path,
        mole_frac=feed_mole_frac,
        mole_flow=feed_flow,
        temp=298.15,  # [K]
        verbose=False,
    )
    solvent = LiquidStream(
        thermo_path,
        mole_frac=solvent_mole_frac,
        mole_flow=solvent_flow,
        temp=298.15,  # [K]
        verbose=False,
    )

    extractor.Phases = phase
    extractor.Inlet = {
        "feed": feed,
        "solvent": solvent,
    }

    di_init = extractor.initialize_model()

    assert extractor.fixed_vals["H_R"] > 0  # [mol]
    assert extractor.fixed_vals["H_E"] > 0  # [mol]
    np.testing.assert_allclose(di_init["x_i"].sum(axis=1), 1.0)
    np.testing.assert_allclose(di_init["y_i"].sum(axis=1), 1.0)
