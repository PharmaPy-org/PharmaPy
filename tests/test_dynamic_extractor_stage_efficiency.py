"""Regression tests for multistage DynamicExtractor efficiency corrections."""

from types import SimpleNamespace

import numpy as np
import pytest

import PharmaPy.DynamicExtraction as dynamic_extraction


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


class _InitializationBatchExtractor:
    """Minimal batch extractor returning fixed extraction states.

    Parameters
    ----------
    k_fun : callable, optional
        Distribution-coefficient callback [-].
    gamma_method : str, optional
        Activity-coefficient method selector [-].
    """
    def __init__(self, k_fun=None, gamma_method="UNIQUAC"):
        self.k_fun = k_fun
        self.gamma_method = gamma_method
        self.result = None

    def solve_unit(self):
        """Store a fixed batch-extraction result.

        Returns
        -------
        None
            The method stores light/heavy moles [mol], densities [mol/L], and
            mole fractions [-] on ``self.result``.
        """
        self.result = SimpleNamespace(
            rho_heavy=np.array(1000.0),  # [mol/L]
            rho_light=np.array(900.0),  # [mol/L]
            mol_light=6.0,  # [mol]
            mol_heavy=4.0,  # [mol]
            x_light=np.array([0.20, 0.35, 0.45]),  # [-]
            x_heavy=np.array([0.25, 0.30, 0.45]),  # [-]
        )


class _InitializationPhase:
    """Minimal liquid phase for initialization residual tests."""
    name_species = ["a", "b", "c"]
    temp = 298.15  # [K]

    def getDensity(self, mole_frac=None, basis="mole", temp=None):
        """Return fixed molar liquid density.

        Parameters
        ----------
        mole_frac : ndarray, optional
            Liquid mole fractions [-].
        basis : str, optional
            Density basis selector [-].
        temp : ndarray, optional
            Liquid temperatures [K].

        Returns
        -------
        ndarray
            Molar density [mol/L].
        """
        mole_frac = np.asarray(mole_frac)  # [-]

        if mole_frac.ndim == 2:
            return np.ones(mole_frac.shape[0]) * 1000.0  # [mol/L]

        return np.array(1000.0)  # [mol/L]

    def getEnthalpy(self, mole_frac, temp, basis="mole"):
        """Return composition-weighted liquid enthalpy.

        Parameters
        ----------
        mole_frac : ndarray
            Liquid mole fractions [-].
        temp : ndarray
            Liquid temperatures [K].
        basis : str, optional
            Enthalpy basis selector [-].

        Returns
        -------
        ndarray
            Liquid enthalpy [J/mol].
        """
        species_enthalpy = np.array([10.0, 20.0, 30.0])  # [J/mol]

        return np.asarray(mole_frac) @ species_enthalpy  # [J/mol]


class _InitializationStream:
    """Minimal inlet stream carrying only density."""
    def __init__(self, density):
        """Create a fixed-density inlet stream.

        Parameters
        ----------
        density : float
            Molar density [mol/L].
        """
        self.density = density  # [mol/L]

    def getDensity(self, basis="mole"):
        """Return the stream molar density.

        Parameters
        ----------
        basis : str, optional
            Density basis selector [-].

        Returns
        -------
        ndarray
            Molar density [mol/L].
        """
        return np.array(self.density)  # [mol/L]


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


def test_initialize_model_broadcasts_componentwise_distribution_coefficients(
        monkeypatch):
    """Initialization applies component-wise ``K_i`` to every stage."""
    captured = {}

    def _capture_root(fun, x0, args):
        """Capture initialization residuals without solving them.

        Parameters
        ----------
        fun : callable
            Initialization residual function [-].
        x0 : ndarray
            Initial mole-fraction guess [-].
        args : tuple
            Residual arguments: temperatures [K], component moles [mol], and
            holdups [mol].

        Returns
        -------
        types.SimpleNamespace
            Object carrying the unchanged mole-fraction guess [-].
        """
        residuals = fun(x0, *args)  # [-]
        captured["y_residuals"] = residuals.reshape(3, 6)[:, 3:]  # [-]

        return SimpleNamespace(x=x0)

    monkeypatch.setattr(
        dynamic_extraction, "BatchExtractor", _InitializationBatchExtractor)
    monkeypatch.setattr(dynamic_extraction, "root", _capture_root)

    extractor = dynamic_extraction.DynamicExtractor(
        num_stages=3,
        k_fun=_component_distribution,
        eff=0.75,
    )
    extractor.Liquid_1 = _InitializationPhase()
    extractor.name_species = extractor.Liquid_1.name_species
    extractor.num_comp = len(extractor.name_species)  # [-]
    extractor.nomenclature()
    extractor.Inlet = {
        "feed": _InitializationStream(1000.0),  # [mol/L]
        "solvent": _InitializationStream(900.0),  # [mol/L]
    }

    extractor.initialize_model()

    expected_y_residuals = np.array([
        [-0.09, 0.26, 0.0],
        [-0.13, 0.12, 0.0],
        [-0.13, 0.12, 0.0],
    ])  # [-]

    np.testing.assert_allclose(captured["y_residuals"],
                               expected_y_residuals)
