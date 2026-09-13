"""Regression coverage for the MSMPR liquid-phase concentration balance.

``MSMPR.material_balances`` corrects the crystallization sink for the shrinking
liquid volume through a ``c_tank / rho`` term. That correction is a *liquid*
volume effect, so it must use the liquid density, as
``BatchCryst.material_balances`` and ``SemibatchCryst.material_balances``
already do. The crystal (solid) density belongs only in the kinetic mass
transfer rate.

Fixture values use the crystallizer solver's units: raw moments in
[um**n/m**3], moments on a metre basis in [m**n/m**3], mass concentrations in
[kg/m**3], densities in [kg/m**3], growth rate in [um/s], volumetric flow in
[m**3/s], and slurry volume in [m**3]. The two densities are deliberately
unequal (1000 vs 1500 [kg/m**3]) so that substituting one for the other is
visible in the asserted values, and the two species carry different
concentrations so a component mix-up cannot pass.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import MSMPR


pytestmark = pytest.mark.unit


class _Liquid:
    """Liquid phase holding the crystallizing solute."""

    def getDensity(self, temp=None):
        """Return liquid density.

        Parameters
        ----------
        temp : float or None, optional
            Liquid temperature [K]. The constant-density fixture ignores it.

        Returns
        -------
        float
            Liquid density [kg/m**3].
        """
        liquid_density = 1000.0  # [kg/m**3]
        return liquid_density


class _Solid:
    """Crystal phase; ``kv`` is the volumetric shape factor."""

    kv = 0.5  # volumetric shape factor [-]

    def getDensity(self, temp=None):
        """Return crystal density.

        Parameters
        ----------
        temp : float or None, optional
            Crystal temperature [K]. The constant-density fixture ignores it.

        Returns
        -------
        float
            Crystal density [kg/m**3].
        """
        crystal_density = 1500.0  # [kg/m**3]
        return crystal_density


class _Kinetics:
    """Constant growth kinetics, so the transfer rate is hand-computable."""

    def get_kinetics(self, conc, temp, kv, moms):
        """Return constant crystallization rates.

        Parameters
        ----------
        conc : numpy.ndarray
            Liquid-phase mass concentrations [kg/m**3].
        temp : float
            Liquid temperature [K].
        kv : float
            Crystal volumetric shape factor [-].
        moms : numpy.ndarray
            Crystal moments [um**n/m**3].

        Returns
        -------
        tuple of float
            Nucleation rate [#/m**3/s], growth rate [um/s], and dissolution
            rate [um/s], respectively.
        """
        nucleation_rate = 0.0  # [#/m**3/s]
        growth_rate = 20.0  # [um/s]
        dissolution_rate = 0.0  # [um/s]
        return nucleation_rate, growth_rate, dissolution_rate

    def alpha_fn(self, conc):
        """Return the growth-rate impurity factor.

        Parameters
        ----------
        conc : numpy.ndarray
            Liquid-phase mass concentrations [kg/m**3].

        Returns
        -------
        float
            Growth-rate impurity factor [-].
        """
        impurity_factor = 1.0  # [-]
        return impurity_factor


def _build_msmpr():
    """Build an ``MSMPR`` populated with the regression fixture.

    Returns
    -------
    PharmaPy.Crystallizers.MSMPR
        Partially initialized crystallizer configured for the direct material
        balance call.

    Notes
    -----
    Construction bypasses ``__init__`` to isolate the balance from solver
    setup while retaining the real public model method and collaborators.
    """
    crystallizer = MSMPR.__new__(MSMPR)
    crystallizer.num_distr = 4
    crystallizer.num_species = 2
    crystallizer.target_ind = 0
    crystallizer.kron_jtg = np.array([1.0, 0.0])  # [-], target-component selector
    crystallizer.basis = "mass_conc"
    crystallizer.method = "moments"
    crystallizer.rad = 1.0  # nuclei radius [um]
    crystallizer.Liquid_1 = _Liquid()
    crystallizer.Solid_1 = _Solid()
    crystallizer._Kinetics = _Kinetics()

    return crystallizer


# Raw tank moments [um**n/m**3]; mu_3 fixes the solid volume fraction
# kv * mu_3 = 0.5 * 0.2 = 0.1 [-], hence a liquid fraction phi = 0.9 [-].
TANK_MOMENTS_RAW = np.array([1.0e10, 5.0e11, 2.0e12, 2.0e17])
TANK_MOMENTS = TANK_MOMENTS_RAW * (1e-6) ** np.arange(4)  # [m**n/m**3]

# Inlet slurry moments [m**n/m**3], consistent with the inlet liquid fraction
# below: 1 - kv * mu_3_in = 1 - 0.5 * 0.1 = 0.95 [-].
INLET_MOMENTS = np.array([2.0e9, 1.0e5, 0.5, 0.1])

TANK_CONC = np.array([200.0, 50.0])  # [kg/m**3], liquid-phase concentrations
INLET_CONC = np.array([250.0, 60.0])  # [kg/m**3]
INLET_PHI = np.array([0.95, 0.05])  # [-], inlet [liquid, solid] volume fractions

SLURRY_VOL = 1.0e-3  # [m**3]
INLET_VOL_FLOW = 1.0e-5  # [m**3/s], giving an inverse residence time of 0.01 [1/s]
TEMP = 300.0  # [K]

# [[liquid, solid] tank densities, [liquid, solid] inlet densities] [kg/m**3].
DENSITIES = [np.array([1000.0, 1500.0]), np.array([990.0, 1500.0])]

RATE_RTOL = 1.0e-12  # [-], deterministic algebra roundoff allowance
RATE_ATOL = 0.0  # [kg/m**3/s], no absolute slack for the nonzero rates


def test_msmpr_transfer_term_uses_liquid_density():
    """Use liquid density only in the MSMPR volume-shrinkage correction."""
    crystallizer = _build_msmpr()

    # Mixed units by field: volumetric flow [m**3/s], inlet moments
    # [m**n/m**3], and liquid-phase mass concentrations [kg/m**3].
    u_inputs = {
        "Inlet": {"vol_flow": INLET_VOL_FLOW, "mu_n": INLET_MOMENTS},
        "Liquid_1": {"mass_conc": INLET_CONC},
    }

    # Mixed state-derivative units as documented by ``material_balances``;
    # ``transf`` is the crystallization rate [kg/m**3/s].
    dmaterial_dt, transf = crystallizer.material_balances(
        time=0.0,
        params=None,
        u_inputs=u_inputs,
        rhos=DENSITIES,
        mu_n=TANK_MOMENTS,
        distrib=TANK_MOMENTS_RAW,
        mass_conc=TANK_CONC,
        temp=TEMP,
        temp_ht=None,
        vol=SLURRY_VOL,
        phi_in=INLET_PHI,
    )

    # Crystal mass transfer keeps the *solid* density:
    # transf = rho_sol * kv * 3 * growth * mu_2_raw * 1e-18
    #        = 1500 * 0.5 * 3 * 20 * 2.0e12 * 1e-18 = 0.09 [kg/m**3/s].
    expected_transf = 0.09  # [kg/m**3/s]
    np.testing.assert_allclose(
        transf,
        expected_transf,
        rtol=RATE_RTOL,
        atol=RATE_ATOL,
    )

    # Hand-computed liquid-phase balance, all terms in [kg/m**3/s]:
    #   flow_term    = 0.01 * (250*0.95 - 200*0.9,  60*0.95 - 50*0.9)
    #                = (0.575, 0.12)
    #   transf_term  = 0.09 * (1 - 200/1000, 0 - 50/1000) = (0.072, -0.0045)
    #   dcomp_dt     = (flow_term - transf_term) / 0.9
    #                = (0.503, 0.1245) / 0.9
    # Using the solid density (1500) in transf_term instead would give
    # (0.5522222..., 0.1366666...), about 1.2 % low on both components.
    expected_dcomp_dt = np.array(
        [0.55888888888888888, 0.13833333333333333]
    )  # [kg/m**3/s]
    np.testing.assert_allclose(
        dmaterial_dt[crystallizer.num_distr:],
        expected_dcomp_dt,
        rtol=RATE_RTOL,
        atol=RATE_ATOL,
    )
