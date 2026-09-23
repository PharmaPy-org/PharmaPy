"""Regression coverage for the BatchCryst concentration Jacobian block.

Fixture values use the solver's crystallizer units: moments in [um**n],
mass concentrations in [kg/m**3], liquid volume in [m**3], temperature in [K],
growth in [um/s], and densities in [kg/m**3].
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst


pytestmark = pytest.mark.unit


class _Liquid:
    def getDensity(self, temp=None):
        """Return liquid density [kg/m**3]."""
        return 1000.0


class _Solid:
    kv = 2.0  # shape factor [-]

    def getDensity(self, temp=None):
        """Return crystal density [kg/m**3]."""
        return 1500.0


class _Kinetics:
    prim_nucl = 0.0
    sec_nucl = 0.0
    growth = 2.0e12  # [um/s]
    params = {
        "growth": [0.0, 0.0, 1.0],  # last entry is growth exponent [-]
        "nucl_prim": [0.0, 0.0, 1.0],  # last entry is exponent [-]
        "nucl_sec": [0.0, 0.0, 0.0, 0.0],  # exponents are [-]
    }

    def get_solubility(self, temp, conc):
        """Return saturated mass concentration [kg/m**3]."""
        return 0.25


def test_batch_cryst_concentration_jacobian_matches_material_balance():
    crystallizer = BatchCryst.__new__(BatchCryst)
    crystallizer.num_distr = 4
    crystallizer.num_species = 2
    crystallizer.target_ind = 0
    crystallizer.kron_jtg = np.array([1.0, 0.0])
    crystallizer.controls = {
        "temp": {
            "fun": lambda time: 298.15,
            "args": (),
            "kwargs": {},
        }
    }
    crystallizer.Liquid_1 = _Liquid()
    crystallizer.Solid_1 = _Solid()
    crystallizer._Kinetics = _Kinetics()

    moments = np.array([1.0, 2.0, 4.0, 8.0])  # [um**n]
    mass_conc = np.array([0.55, 0.20])  # [kg/m**3]
    vol_liq = 2.0  # [m**3]
    states = np.concatenate((moments, mass_conc, [vol_liq]))

    jacobian = crystallizer.jac_states(
        time=0.0,
        states=states,
        params=None,
        return_only=False,
    )

    # Hand-computed from the fixture values: tr = 0.072 [kg/s],
    # dtr_dconc_tg = 0.24 [m**3/s], and tr / rho_l = 7.2e-5 [m**3/s].
    expected_conc_block = np.array([
        [-0.119898, 0.0],
        [2.4e-5, 3.6e-5],
    ])  # [1/s]

    np.testing.assert_allclose(
        jacobian[crystallizer.num_distr:-1, crystallizer.num_distr:-1],
        expected_conc_block,
        rtol=1e-12,
        atol=1e-12,
    )
