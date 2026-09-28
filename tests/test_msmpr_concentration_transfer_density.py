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
[m**3/s], and slurry volume in [m**3]. The real database densities at 300 K
are deliberately unequal (about 996 for the liquid and 1230 [kg/m**3] for the
solid), so substituting one for the other is visible in the asserted values.
The target and second species carry different
concentrations, while two database padding species remain at zero, so a
component mix-up cannot pass.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import MSMPR
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Slurry
from PharmaPy.Phases import LiquidPhase, SolidPhase


pytestmark = pytest.mark.unit


def _build_msmpr(data_path):
    """Build an ``MSMPR`` populated with the regression fixture.

    Parameters
    ----------
    data_path : dict
        Repository test-data paths.

    Returns
    -------
    PharmaPy.Crystallizers.MSMPR
        Partially initialized crystallizer configured for the direct material
        balance call.

    Notes
    -----
    The public constructors configure real phase, slurry, and kinetics
    collaborators without invoking the optional solver backend.
    """
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    liquid = LiquidPhase(
        thermo_path,
        temp=TEMP,
        vol=SLURRY_VOL,  # [m**3]
        mass_frac=np.array([0.4, 0.1, 0.1, 0.4]),  # [-]
        verbose=False,
    )
    solid = SolidPhase(
        thermo_path,
        temp=TEMP,
        moments=TANK_MOMENTS,
        mass_frac=np.array([1.0, 0.0, 0.0, 0.0]),  # [-]
        kv=0.5,  # [-]
    )
    slurry = Slurry(vol=SLURRY_VOL, moments=TANK_MOMENTS)
    slurry.Phases = [liquid, solid]

    crystallizer = MSMPR(
        "A",
        method="moments",
        vol_tank=SLURRY_VOL,
        adiabatic=True,
        rad_zero=1.0,  # [um]
    )
    crystallizer.Phases = slurry
    crystallizer.num_species = liquid.num_species  # [-]

    kinetics = CrystKinetics(
        coeff_solub=[0.0],  # [kg/m**3]
        growth=[20.0, 0.0, 0.0],  # [um/s], [J/mol], [-]
        sup_sat_type="absolute",
    )
    kinetics.target_idx = crystallizer.target_ind
    crystallizer.Kinetics = kinetics

    return crystallizer


# Raw tank moments [um**n/m**3]; mu_3 fixes the solid volume fraction
# kv * mu_3 = 0.5 * 0.2 = 0.1 [-], hence a liquid fraction phi = 0.9 [-].
TANK_MOMENTS_RAW = np.array([1.0e10, 5.0e11, 2.0e12, 2.0e17])
TANK_MOMENTS = TANK_MOMENTS_RAW * (1e-6) ** np.arange(4)  # [m**n/m**3]

# Inlet slurry moments [m**n/m**3], consistent with the inlet liquid fraction
# below: 1 - kv * mu_3_in = 1 - 0.5 * 0.1 = 0.95 [-].
INLET_MOMENTS = np.array([2.0e9, 1.0e5, 0.5, 0.1])

TANK_CONC = np.array([
    200.0, 50.0, 0.0, 0.0
])  # [kg/m**3], liquid-phase concentrations
INLET_CONC = np.array([250.0, 60.0, 0.0, 0.0])  # [kg/m**3]
INLET_PHI = np.array([0.95, 0.05])  # [-], inlet [liquid, solid] volume fractions

SLURRY_VOL = 1.0e-3  # [m**3]
INLET_VOL_FLOW = 1.0e-5  # [m**3/s], giving an inverse residence time of 0.01 [1/s]
TEMP = 300.0  # [K]

RATE_RTOL = 1.0e-12  # [-], deterministic algebra roundoff allowance
RATE_ATOL = 0.0  # [kg/m**3/s], no absolute slack for the nonzero rates


def test_msmpr_transfer_term_uses_liquid_density(data_path):
    """Use liquid density only in the MSMPR volume-shrinkage correction."""
    crystallizer = _build_msmpr(data_path)
    liquid_density = crystallizer.Liquid_1.getDensity(temp=TEMP)  # [kg/m**3]
    solid_density = crystallizer.Solid_1.getDensity(temp=TEMP)  # [kg/m**3]
    densities = [
        np.array([liquid_density, solid_density]),
        np.array([liquid_density, solid_density]),
    ]  # [kg/m**3]

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
        rhos=densities,
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
    # All factors below come from the real solid phase and stated kinetic
    # fixture, so the result follows the database density at 300 K.
    expected_transf = (
        solid_density * crystallizer.Solid_1.kv * 3.0
        * 20.0 * TANK_MOMENTS_RAW[2] * 1.0e-18
    )  # [kg/m**3/s]
    np.testing.assert_allclose(
        transf,
        expected_transf,
        rtol=RATE_RTOL,
        atol=RATE_ATOL,
    )

    # Hand-computed liquid-phase balance, all terms in [kg/m**3/s]:
    #   flow_term    = 0.01 * (250*0.95 - 200*0.9,  60*0.95 - 50*0.9)
    #                = (0.575, 0.12)
    #   transf_term  = expected_transf
    #                  * (1 - 200/liquid_density,
    #                     0 - 50/liquid_density)
    #   dcomp_dt     = (flow_term - transf_term) / 0.9
    # Using ``solid_density`` in the concentration correction instead would
    # change both component rates; the assertion below pins the liquid basis.
    flow_term = (
        INLET_VOL_FLOW / SLURRY_VOL
        * (INLET_CONC * INLET_PHI[0] - TANK_CONC * 0.9)
    )  # [kg/m**3/s]
    transfer_term = expected_transf * (
        crystallizer.kron_jtg - TANK_CONC / liquid_density
    )  # [kg/m**3/s]
    expected_dcomp_dt = (flow_term - transfer_term) / 0.9  # [kg/m**3/s]
    np.testing.assert_allclose(
        dmaterial_dt[crystallizer.num_distr:],
        expected_dcomp_dt,
        rtol=RATE_RTOL,
        atol=RATE_ATOL,
    )
