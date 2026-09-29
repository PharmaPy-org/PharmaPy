"""Regression tests for DynamicExtractor ideal-equilibrium behavior.

Issue #212 tracks restoring non-ideal callback and initialization-handoff
coverage once sourced activity-model parameters are available.
"""

import numpy as np
import pytest

import PharmaPy.DynamicExtraction as dynamic_extraction
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


def test_default_k_fun_is_unity_under_the_ideal_model(data_path):
    """The real ideal activity model produces unit distribution factors.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.

    Notes
    -----
    An ideal model cannot cover ratio direction, composition dependence, or
    ``gamma_model`` forwarding because every activity coefficient is one.
    Issue #212 tracks those non-ideal regressions; no checked-in database yet
    carries the sourced UNIQUAC or UNIFAC parameters they require.
    """
    database_path = data_path["flowsheet"] / "compound_database.json"
    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=2,
        gamma_model="ideal",
    )
    extractor.Phases = LiquidPhase(
        str(database_path),
        moles=10.0,  # [mol]
        mole_frac=np.array([0.20, 0.20, 0.20, 0.20, 0.20]),  # [-]
        temp=298.15,  # [K]
        pres=101325.0,  # [Pa]
    )

    x_light = np.array([
        [0.20, 0.10, 0.25, 0.15, 0.30],
        [0.10, 0.20, 0.15, 0.25, 0.30],
    ])  # [-]
    x_heavy = np.array([
        [0.15, 0.20, 0.10, 0.25, 0.30],
        [0.25, 0.10, 0.20, 0.15, 0.30],
    ])  # [-]
    temp = np.array([298.15, 301.15])  # [K]

    k_i = extractor.k_fun(x_light, x_heavy, temp)  # K_i [-]

    np.testing.assert_allclose(k_i, np.ones_like(x_light))


def test_default_k_fun_rejects_unknown_activity_model():
    """DynamicExtractor rejects unknown default activity-model selectors."""
    # Lowercase ``uniquac`` is deliberate: activity-model selectors are
    # case-sensitive and must match the exact Phases.getActivityCoeff branch.
    with pytest.raises(ValueError, match="gamma_model must be one of"):
        dynamic_extraction.DynamicExtractor(
            num_stages=1,
            gamma_model="uniquac",
        )


def test_initialize_model_produces_normalized_positive_ideal_state(data_path):
    """Real ideal-phase initialization yields normalized positive states.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.

    Notes
    -----
    Under the ideal model and unit stage efficiency, ``x_i == y_i`` is an
    identity and does not cover the callback handoff to ``BatchExtractor``.
    Issue #212 tracks restoring that handoff assertion with sourced non-ideal
    data.
    """
    database_path = data_path["flowsheet"] / "compound_database.json"
    # Representative extraction pair: the feed is A-rich and the solvent
    # stream is enriched in the database's designated solvent component.
    feed_mole_fraction = np.array([0.35, 0.20, 0.15, 0.15, 0.15])  # [-]
    solvent_mole_fraction = np.array([0.05, 0.20, 0.25, 0.25, 0.25])  # [-]
    # Non-uniform holdup makes the ideal-model identity composition-specific
    # instead of permutation-invariant.
    holdup_mole_fraction = np.array(
        [0.30, 0.25, 0.20, 0.15, 0.10]
    )  # [-]
    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=1,
        gamma_model="ideal",
    )
    extractor.Phases = LiquidPhase(
        str(database_path),
        moles=10.0,  # [mol]
        mole_frac=holdup_mole_fraction,
        temp=298.15,  # [K]
        pres=101325.0,  # [Pa]
    )
    extractor.Inlet = {
        "feed": LiquidStream(
            str(database_path),
            mole_flow=5.0,  # [mol/s]
            mole_frac=feed_mole_fraction,
            temp=298.15,  # [K]
            pres=101325.0,  # [Pa]
        ),
        "solvent": LiquidStream(
            str(database_path),
            mole_flow=3.0,  # [mol/s]
            mole_frac=solvent_mole_fraction,
            temp=298.15,  # [K]
            pres=101325.0,  # [Pa]
        ),
    }

    initial_states = extractor.initialize_model()

    np.testing.assert_allclose(initial_states["x_i"].sum(axis=1), 1.0)
    np.testing.assert_allclose(initial_states["y_i"].sum(axis=1), 1.0)
    np.testing.assert_allclose(initial_states["x_i"], initial_states["y_i"])
    expected_holdup = np.tile(
        holdup_mole_fraction, (extractor.num_stages, 1)
    )  # [-]
    np.testing.assert_allclose(initial_states["x_i"], expected_holdup)
    assert extractor.fixed_vals["H_R"] > 0.0  # [mol]
    assert extractor.fixed_vals["H_E"] > 0.0  # [mol]
