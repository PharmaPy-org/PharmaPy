"""Regression tests for multistage DynamicExtractor efficiency corrections."""

import numpy as np
import pytest

import PharmaPy.DynamicExtraction as dynamic_extraction
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


_K_STAGE = np.array([
    [0.60, 1.20, 1.50],
    [1.80, 0.75, 1.05],
    [0.90, 1.40, 0.80],
])  # [-]


def _stage_distribution(x_light, x_heavy, temp):
    """Return stage-specific distribution coefficients.

    Parameters
    ----------
    x_light, x_heavy : ndarray
        Light- and heavy-phase mole fractions [-].
    temp : ndarray
        Stage temperatures [K].

    Returns
    -------
    ndarray
        Stage-specific distribution coefficients ``K_i`` [-].
    """
    num_stages = np.shape(x_light)[0]  # [-]

    return np.broadcast_to(_K_STAGE[:num_stages], np.shape(x_light))  # [-]


def _component_distribution(x_light, x_heavy, temp):
    """Return component-wise distribution coefficients.

    Parameters
    ----------
    x_light, x_heavy : ndarray
        Light- and heavy-phase mole fractions [-].
    temp : ndarray
        Stage temperatures [K].

    Returns
    -------
    ndarray
        Component distribution coefficients ``K_i`` [-].
    """
    return np.array([0.60, 1.20, 1.50])  # [-]


def _invalid_distribution(x_light, x_heavy, temp):
    """Return an invalid number of distribution coefficients.

    Parameters
    ----------
    x_light, x_heavy : ndarray
        Light- and heavy-phase mole fractions [-].
    temp : ndarray
        Stage temperatures [K].

    Returns
    -------
    ndarray
        Distribution coefficients with the wrong component count [-].
    """
    return np.array([0.60, 1.20])  # [-]


def _initialization_distribution(x_light, x_heavy, temp):
    """Return component-wise coefficients that form two liquid phases.

    Parameters
    ----------
    x_light, x_heavy : ndarray
        Light- and heavy-phase mole fractions [-].
    temp : ndarray
        Stage temperatures [K].

    Returns
    -------
    ndarray
        Component distribution coefficients ``K_i`` [-].

    Notes
    -----
    The coefficients are a test-design assumption spanning both sides of one.
    With the representative feed below, the Rachford-Rice endpoint residuals
    have opposite signs, guaranteeing a two-liquid-phase batch seed before
    the real staged root solve.
    """
    return np.array([0.50, 0.80, 1.30, 2.00, 1.00])  # [-]


def _three_stage_material_inputs():
    """Build the staged material-balance fixture.

    Returns
    -------
    dict
        Material-balance arguments. Mole fractions and distribution
        coefficients are dimensionless [-], temperatures are [K], flows are
        [mol/s], holdups are [mol], and internal energies are [J].
    """
    x_i = np.array([
        [0.20, 0.35, 0.45],
        [0.32, 0.28, 0.40],
        [0.27, 0.31, 0.42],
    ])  # [-]
    y_i = np.array([
        [0.25, 0.30, 0.45],
        [0.18, 0.42, 0.40],
        [0.21, 0.34, 0.45],
    ])  # [-]
    x_in = np.array([0.30, 0.30, 0.40])  # [-]
    y_in = np.array([0.22, 0.38, 0.40])  # [-]
    x_augm = np.vstack((x_in, x_i))  # [-]
    y_augm = np.vstack((y_i, y_in))  # [-]
    temp_augm = np.array([296.15, 298.15, 299.15, 300.15, 301.15])  # [K]
    light_flows = np.array([5.0, 4.0, 3.5, 3.0])  # [mol/s]
    heavy_flows = np.array([2.5, 3.0, 3.2, 2.8])  # [mol/s]

    return {
        "x_i": x_i,
        "y_i": y_i,
        "holdup_light": np.array([10.0, 14.0, 12.0]),  # [mol]
        "holdup_heavy": np.array([4.0, 9.0, 6.0]),  # [mol]
        "u_int": np.array([0.0, 0.0, 0.0]),  # [J]
        "temp": np.array([298.15, 299.15, 300.15]),  # [K]
        "di_sdot": None,
        "augm_arrays": (
            x_augm, y_augm, temp_augm, light_flows, heavy_flows
        ),
    }


def test_material_balances_apply_recursive_efficiency_correction():
    """Later-stage correction propagates upstream corrected rates."""
    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=3,
        k_fun=_stage_distribution,
        eff=0.75,
    )

    material_residuals, equilibrium_residuals = extractor.material_balances(
        0.0, **_three_stage_material_inputs())

    expected_material = np.array([
        [0.0465909090909091, 0.03719512195121951, 0.0],
        [0.00178626149131767, 0.0144212619300106, 0.0],
        [0.0133966286814777, 0.00286118678465645, 0.0],
    ])  # first two columns [1/s], final column [-]
    expected_equilibrium = np.array([
        [-0.1500, 0.1400, 0.0],
        [0.4680, -0.2275, 0.0],
        [0.0180, 0.1080, 0.0],
    ])  # [-]
    # Stages 2 and 3 include
    # H_E,j * (K_i,j / eff) * (1 - eff) / div_j * dx_i,j-1/dt.
    # The unequal K_i and H_E values guard against using stage 1 quantities or
    # the uncorrected upstream rate.

    np.testing.assert_allclose(material_residuals, expected_material)
    np.testing.assert_allclose(equilibrium_residuals, expected_equilibrium)


def test_material_balances_skip_efficiency_correction_at_full_efficiency():
    """Full efficiency leaves multistage rates at ``RHS_j / div_j``."""
    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=3,
        k_fun=_stage_distribution,
        eff=1.0,
    )

    material_residuals, equilibrium_residuals = extractor.material_balances(
        0.0, **_three_stage_material_inputs())

    expected_material = np.array([
        [0.04959677419354838, 0.0412162162162162, 0.0],
        [-0.00622516556291391, 0.01195180722891566, 0.0],
        [0.01459770114942529, 0.00127450980392157, 0.0],
    ])  # first two columns [1/s], final column [-]
    expected_equilibrium = np.array([
        [-0.1300, 0.1200, 0.0],
        [0.3960, -0.2100, 0.0],
        [0.0330, 0.0940, 0.0],
    ])  # [-]

    np.testing.assert_allclose(material_residuals, expected_material)
    np.testing.assert_allclose(equilibrium_residuals, expected_equilibrium)


def test_material_balances_accept_componentwise_distribution_coefficients():
    """Component-wise ``K_i`` values broadcast across stages."""
    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=3,
        k_fun=_component_distribution,
        eff=0.75,
    )

    material_residuals, equilibrium_residuals = extractor.material_balances(
        0.0, **_three_stage_material_inputs())

    expected_material = np.array([
        [0.0465909090909091, 0.03719512195121951, 0.0],
        [-0.00491209262435678, 0.01344726897973205, 0.0],
        [0.01476818386016499, 0.00269784470145171, 0.0],
    ])  # first two columns [1/s], final column [-]
    expected_equilibrium = np.array([
        [-0.1500, 0.1400, 0.0],
        [0.0360, -0.1120, 0.0],
        [-0.0580, 0.0440, 0.0],
    ])  # [-]

    np.testing.assert_allclose(material_residuals, expected_material)
    np.testing.assert_allclose(equilibrium_residuals, expected_equilibrium)


def test_material_balances_reject_unbroadcastable_distribution_coefficients():
    """Invalid ``K_i`` shape raises a specific callback contract error."""
    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=3,
        k_fun=_invalid_distribution,
        eff=0.75,
    )

    with pytest.raises(
            ValueError,
            match="k_fun must return distribution coefficients broadcastable"):
        extractor.material_balances(0.0, **_three_stage_material_inputs())


def test_initialize_model_solves_componentwise_distribution_coefficients(
        data_path):
    """Real phase and root collaborators solve every staged ``K_i`` row."""
    database_path = data_path["flowsheet"] / "compound_database.json"
    # Representative extraction pair: the feed is A-rich and the solvent
    # stream is enriched in the database's designated solvent component.
    feed_mole_fraction = np.array([0.35, 0.20, 0.15, 0.15, 0.15])  # [-]
    solvent_mole_fraction = np.array([0.05, 0.20, 0.25, 0.25, 0.25])  # [-]
    holdup_mole_fraction = (
        feed_mole_fraction + solvent_mole_fraction
    ) / 2  # [-]
    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=3,
        k_fun=_initialization_distribution,
        eff=0.75,
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

    x_light = initial_states["x_i"]  # [-]
    x_heavy = initial_states["y_i"]  # [-]
    k_stage = np.broadcast_to(
        _initialization_distribution(x_light, x_heavy, initial_states["temp"]),
        x_light.shape,
    )  # [-]
    equilibrium_residuals = np.zeros_like(x_heavy)  # [-]
    equilibrium_residuals[0] = (
        k_stage[0] / extractor.eff * x_light[0] - x_heavy[0]
    )  # [-]
    equilibrium_residuals[1:] = (
        k_stage[1:] / extractor.eff
        * (x_light[1:] - x_light[:-1] * (1 - extractor.eff))
        - x_heavy[1:]
    )  # [-]
    equilibrium_residuals[:, -1] = x_heavy.sum(axis=1) - 1  # [-]
    component_moles = np.tile(
        extractor.Liquid_1.moles * holdup_mole_fraction,
        (extractor.num_stages, 1),
    )  # [mol]

    assert extractor.target_states["light_phase"] == "solvent"
    assert extractor.target_states["heavy_phase"] == "feed"
    assert np.all(x_light > 0.0)
    assert np.all(x_heavy > 0.0)
    np.testing.assert_allclose(x_light.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(x_heavy.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(equilibrium_residuals, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        extractor.fixed_vals["H_R"] * x_light
        + extractor.fixed_vals["H_E"] * x_heavy,
        component_moles,
        atol=1e-12,
    )
