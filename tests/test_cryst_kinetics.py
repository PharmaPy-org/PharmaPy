"""Crystallization kinetics regressions for scalar target states."""

import numpy as np
import pytest

from PharmaPy.Crystallizers import MSMPR
from PharmaPy.Kinetics import CrystKinetics


pytestmark = pytest.mark.unit


def _primary_growth_kinetics():
    """Create kinetics with deterministic primary nucleation and growth.

    Returns
    -------
    CrystKinetics
        Kinetics object using concentration-like inputs [kg/m**3],
        temperature [K], normalized fixture coefficients [-], primary
        nucleation [#/m**3/s], and growth [um/s].
    """
    kin = CrystKinetics(
        coeff_solub=[0.1, 0.0, 0.0],  # [kg/m**3]
        nucl_prim=[1.0, 0.0, 1.0],  # [-], normalized fixture parameters
        growth=[1.0, 0.0, 1.0],  # [-], normalized fixture parameters
        sup_sat_type="absolute",
    )
    kin.target_idx = 0  # [-]
    return kin


@pytest.mark.parametrize("conc", [0.5, np.array(0.5)])
def test_get_kinetics_accepts_scalar_target_concentration(conc):
    """Scalar target concentrations return scalar kinetic rates.

    Parameters
    ----------
    conc : float or ndarray
        Target concentration fixture [kg/m**3].
    """
    kin = _primary_growth_kinetics()

    nucl, growth, dissol = kin.get_kinetics(
        conc, 298.15, 0.5)  # [#/m**3/s], [um/s], [um/s]

    assert np.ndim(nucl) == 0
    assert np.ndim(growth) == 0
    assert np.ndim(dissol) == 0
    assert nucl == pytest.approx(0.4)
    assert growth == pytest.approx(0.4)
    assert dissol == pytest.approx(0.0)


def test_get_kinetics_preserves_vector_target_selection():
    """Vector concentration inputs still select the configured target index."""
    kin = _primary_growth_kinetics()
    kin.target_idx = 1  # [-]

    conc = np.array([[9.0, 0.5], [8.0, 0.6]])  # [kg/m**3]
    temp = np.array([298.15, 298.15])  # [K]
    moments = np.zeros((2, 4))  # [-], unused normalized moment fixture

    nucl, growth, dissol = kin.get_kinetics(
        conc, temp, 0.5, moments)  # [#/m**3/s], [um/s], [um/s]

    np.testing.assert_allclose(nucl, [0.4, 0.5])
    np.testing.assert_allclose(growth, [0.4, 0.5])
    np.testing.assert_allclose(dissol, [0.0, 0.0])


def test_get_kinetics_uses_secondary_parameters_from_vector_update():
    """Vector parameter updates drive secondary nucleation rates."""
    kin = _primary_growth_kinetics()
    kin.set_params(
        np.array([
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
        ])  # [-], normalized fixture parameters
    )

    conc = np.array([[0.5], [0.6]])  # [kg/m**3]
    temp = np.array([298.15, 298.15])  # [K]
    moments = np.full((2, 4), 2.0)  # [-], normalized moment fixture

    prim, sec, growth, dissol = kin.get_kinetics(
        conc, temp, 0.5, moments, nucl_sec_out=True
    )  # [#/m**3/s], [#/m**3/s], [um/s], [um/s]

    np.testing.assert_allclose(prim, [0.4, 0.5])
    np.testing.assert_allclose(sec, [0.4, 0.5])
    np.testing.assert_allclose(growth, [0.4, 0.5])
    np.testing.assert_allclose(dissol, [0.0, 0.0])


def test_msmpr_steady_state_accepts_scalar_seed():
    """MSMPR steady-state solve accepts a scalar seed fraction."""

    crystallizer = MSMPR.__new__(MSMPR)
    crystallizer.basis = 'mass_frac'
    crystallizer.method = 'moments'
    crystallizer.rad = 0.0  # [um], zero-size nuclei
    crystallizer.kron_jtg = np.array([1.0])  # [-], one target species
    crystallizer.num_distr = 4  # zeroth through third population moments
    crystallizer.vol_slurry = 1.0  # [m**3]
    crystallizer.target_ind = 0  # [-]
    crystallizer._Kinetics = _primary_growth_kinetics()

    class Solid:
        """Solid-phase fixture for MSMPR steady-state solve."""

        x_distrib = np.array([0.0, 1.0])  # [um]
        temp = 298.15  # [K], initial phase temperature
        kv = 0.0  # [-]
        num_mom = 4  # zeroth through third moments

        def getDensity(self, temp):
            """Return a constant solid density.

            Parameters
            ----------
            temp : float
                Temperature [K].

            Returns
            -------
            float
                Solid density [kg/m**3].
            """
            return 1.0  # [kg/m**3]

    class Liquid:
        """Liquid-phase fixture with one mass fraction."""

        temp = 298.15  # [K], initial phase temperature
        mass_frac = np.array([0.5])  # [-]
        mass_conc = np.array([0.5])  # [kg/m**3], density is 1 kg/m**3

        def getDensity(self, temp=None):
            """Return the synthetic liquid density.

            Parameters
            ----------
            temp : float or None, optional
                Liquid temperature [K]; omitted for the constant-density RHS.

            Returns
            -------
            float
                Constant liquid density [kg/m**3].
            """
            return 1.0  # [kg/m**3], immaterial when kv=0

    class Inlet:
        """Inlet fixture for the MSMPR steady-state solve."""

        vol_flow = 1.0  # [m**3/s]
        Liquid_1 = Liquid()

    crystallizer.Solid_1 = Solid()
    crystallizer._Inlet = Inlet()
    crystallizer.Liquid_1 = crystallizer.Inlet.Liquid_1

    x_vec, f_convg, w_convg, info, final_fn = crystallizer.solve_steady_state(
        0.3, 298.15)  # [um], [#/m**3/um], [-], [-], [-]

    np.testing.assert_allclose(x_vec, [0.0, 1.0])
    np.testing.assert_allclose(f_convg, [1.0, np.exp(-2.5)])
    assert w_convg == pytest.approx(0.5)
    assert info.converged
    assert final_fn == pytest.approx(0.0)
