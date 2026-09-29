"""Regression coverage for the BatchCryst concentration Jacobian block.

Fixture values use the solver's crystallizer units: moments in [um**n],
mass concentrations in [kg/m**3], liquid volume in [m**3], temperature in [K],
growth in [um/s], and densities in [kg/m**3].
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Slurry
from PharmaPy.Phases import LiquidPhase, SolidPhase


pytestmark = pytest.mark.unit


def test_batch_cryst_concentration_jacobian_matches_material_balance(data_path):
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    temperature = 298.15  # [K]
    liquid = LiquidPhase(
        thermo_path,
        temp=temperature,
        vol=2.0,  # [m**3]
        mass_frac=np.array([0.55, 0.20, 0.10, 0.15]),  # [-]
        verbose=False,
    )
    solid = SolidPhase(
        thermo_path,
        temp=temperature,
        moments=np.array([1.0, 0.0, 0.0, 1.0e-6]),  # [m**n]
        mass_frac=np.array([1.0, 0.0, 0.0, 0.0]),  # [-]
        kv=2.0,  # [-]
    )
    slurry = Slurry()
    slurry.Phases = [liquid, solid]

    crystallizer = BatchCryst(
        "A",
        method="moments",
        controls={"temp": lambda time: temperature},
        adiabatic=True,
    )
    crystallizer.Phases = slurry
    crystallizer.num_species = liquid.num_species  # [-]

    growth_rate = 2.0e12  # [um/s]
    saturated_concentration = 0.25  # [kg/m**3]
    target_concentration = 0.55  # [kg/m**3]
    kinetics = CrystKinetics(
        coeff_solub=[saturated_concentration],  # [kg/m**3]
        growth=[
            growth_rate / (target_concentration - saturated_concentration),
            0.0,
            1.0,
        ],  # [um/s], [J/mol], [-]
        sup_sat_type="absolute",
    )
    kinetics.target_idx = crystallizer.target_ind
    crystallizer.Kinetics = kinetics

    moments = np.array([1.0, 2.0, 4.0, 8.0])  # [um**n]
    mass_conc = np.array([0.55, 0.20, 0.10, 0.15])  # [kg/m**3]
    vol_liq = 2.0  # [m**3]
    states = np.concatenate((moments, mass_conc, [vol_liq]))

    kinetics.get_kinetics(
        mass_conc,
        temperature,
        solid.kv,
        moments,
    )

    jacobian = crystallizer.jac_states(
        time=0.0,
        states=states,
        params=None,
        return_only=False,
    )

    liquid_density = liquid.getDensity(temp=temperature)  # [kg/m**3]
    solid_density = solid.getDensity(temp=temperature)  # [kg/m**3]
    transfer_rate = (
        3.0 * solid.kv * kinetics.growth * moments[2] * solid_density
        * (1.0e-6) ** 3
    )  # [kg/s]
    transfer_derivative = transfer_rate / (
        target_concentration - saturated_concentration
    )  # [m**3/s]
    target_selector = crystallizer.kron_jtg  # [-]
    first_term = np.outer(
        target_selector - mass_conc / liquid_density,
        target_selector,
    )  # [-]
    volume_term = -transfer_rate / liquid_density * np.eye(
        len(mass_conc)
    )  # [m**3/s]
    expected_conc_block = -(
        transfer_derivative * first_term + volume_term
    ) / vol_liq  # [1/s]

    np.testing.assert_allclose(
        jacobian[crystallizer.num_distr:-1, crystallizer.num_distr:-1],
        expected_conc_block,
        rtol=1e-12,
        atol=1e-12,
    )
