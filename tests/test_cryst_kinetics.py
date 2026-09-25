"""Crystallization kinetics regressions for scalar target states."""

import numpy as np
import pytest

from PharmaPy.Crystallizers import MSMPR
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream


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


def test_msmpr_steady_state_accepts_scalar_seed(data_path):
    """A real, positive-volume crystal population accepts a scalar seed.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic data paths.

    Notes
    -----
    The synthetic primary prefactor gives B/G = 1e15 #/m**3/um. With a
    one-second residence time and growth near 0.4 um/s, the exponential
    population has third moment about 6 * 1e15 * 0.4**4 * 1e-18 m**3/m**3.
    This resolves nonzero crystal volume with the positive phase shape factor
    required by #165/#263, instead of bypassing that boundary with kv=0.
    """
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    size_grid = np.array([0.0, 1.0])  # [um]
    distribution = np.zeros(2)  # [#/um]
    liquid_mass_fraction = np.array([0.5, 0.5, 0.0, 0.0])  # [-]
    solid_mass_fraction = np.array([1.0, 0.0, 0.0, 0.0])  # [-]

    liquid = LiquidPhase(
        thermo_path,
        temp=298.15,  # [K]
        vol=1.0,  # [m**3]
        mass_frac=liquid_mass_fraction,
        verbose=False,
    )
    solid = SolidPhase(
        thermo_path,
        temp=298.15,  # [K]
        x_distrib=size_grid,
        distrib=distribution,
        mass_frac=solid_mass_fraction,
    )
    slurry = Slurry(vol=1.0, x_distrib=size_grid, distrib=distribution)
    slurry.Phases = [liquid, solid]

    inlet_liquid = LiquidStream(
        thermo_path,
        temp=298.15,  # [K]
        mass_frac=liquid_mass_fraction,
        verbose=False,
    )
    inlet_solid = SolidStream(
        thermo_path,
        temp=298.15,  # [K]
        x_distrib=size_grid,
        distrib=distribution,
        mass_frac=solid_mass_fraction,
    )
    inlet = SlurryStream(
        vol_flow=1.0,  # [m**3/s]
        x_distrib=size_grid,
        distrib=distribution,
    )
    inlet.Phases = [inlet_liquid, inlet_solid]

    crystallizer = MSMPR(
        "A",
        method="moments",
        basis="mass_frac",
        vol_tank=1.0,  # [m**3]
        adiabatic=True,
    )
    crystallizer.Phases = slurry
    crystallizer.Kinetics = CrystKinetics(
        coeff_solub=[0.1, 0.0, 0.0],  # [-], constant solubility mass fraction
        nucl_prim=[1e15, 0.0, 1.0],  # [#/m**3/s], [J/mol], [-]
        growth=[1.0, 0.0, 1.0],  # [um/s], [J/mol], [-]
        sup_sat_type="absolute",
    )
    crystallizer.Kinetics.target_idx = crystallizer.target_ind
    crystallizer.Inlet = inlet

    x_vec, f_convg, w_convg, info, final_fn = crystallizer.solve_steady_state(
        0.3, 298.15)  # [um], [#/m**3/um], [-], [-], [-]

    np.testing.assert_allclose(x_vec, [0.0, 1.0])
    expected_population = 1e15 * np.exp(
        -size_grid / (w_convg - 0.1))  # [#/m**3/um], B/G * exp(-x/(G*tau))
    np.testing.assert_allclose(f_convg, expected_population)
    assert 0.1 < w_convg < liquid_mass_fraction[0]
    assert info.converged
    assert final_fn == pytest.approx(0.0)
