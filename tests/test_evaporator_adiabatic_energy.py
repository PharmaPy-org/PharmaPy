"""Real-phase drum energy includes vapor outflow and condensate reflux."""

import numpy as np
import pytest

from PharmaPy.Evaporators import ContinuousEvaporator
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "reflux_ratio",
    [
        0.0,  # [-]
        0.25,  # [-]
    ],
)
def test_adiabatic_energy_residual_includes_vapor_enthalpy(
        data_path, reflux_ratio):
    """Balance feed, liquid, gross vapor, and returned condensate enthalpies.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic data paths.
    reflux_ratio : float
        Fraction of gross vapor returned as liquid condensate [-].

    Notes
    -----
    PR #263 corrected the drum boundary: all gross vapor leaves with vapor
    enthalpy and only the reflux fraction returns with condensate enthalpy.
    Adiabatic excludes jacket heat, not the separate condenser's duty.
    """
    # Enthalpies are J/mol; flows are mol/s, amounts are mol, pressure is Pa,
    # and volume is m^3. Thus the two residuals are J/s and J, respectively.
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    composition = np.array([1.0, 0.0, 0.0, 0.0])  # [-]
    evaporator = ContinuousEvaporator(
        vol_drum=2.0,  # [m**3]
        adiabatic=True,
        reflux_ratio=reflux_ratio,
    )
    evaporator.Phases = LiquidPhase(
        thermo_path,
        temp=350.0,  # [K]
        moles=3.0,  # [mol]
        mole_frac=composition,
        verbose=False,
    )
    evaporator.Inlet = LiquidStream(
        thermo_path,
        temp=300.0,  # [K]
        mole_flow=4.0,  # [mol/s]
        mole_frac=composition,
        verbose=False,
    )

    result = evaporator.energy_balances(  # [J/s, J]
        time=0.0,
        flow_liq=1.0,
        flow_vap=2.0,
        vol_liq=1.0,
        u_int=40.0,
        temp=350.0,
        x_liq=composition,
        y_vap=composition,
        mol_i=np.array([3.0, 0.0, 0.0, 0.0]),  # [mol]
        mol_liq=3.0,
        mol_vap=4.0,
        pres=101325.0,
        u_inputs={
            "mole_flow": 4.0,
            "mole_frac": composition,
            "temp": 300.0,
        },
    )

    h_in = evaporator.Inlet.getEnthalpy(
        temp=300.0, mole_frac=composition, basis="mole"
    )  # [J/mol]
    h_liq = evaporator.Liquid_1.getEnthalpy(
        temp=350.0, mole_frac=composition, basis="mole"
    )  # [J/mol]
    h_vap = evaporator.Vapor_1.getEnthalpy(
        temp=350.0, mole_frac=composition, basis="mole"
    )  # [J/mol]
    if reflux_ratio == 0:
        h_top = h_vap  # [J/mol]
    else:
        bubble_temp = evaporator.Liquid_1.getBubblePoint(
            101325.0, mole_frac=composition
        )  # [K]
        h_top = evaporator.Liquid_1.getEnthalpy(
            temp=bubble_temp, mole_frac=composition, basis="mole"
        )  # [J/mol]

    expected_energy_rate = (
        4.0 * h_in - 1.0 * h_liq
        - 2.0 * h_vap + reflux_ratio * 2.0 * h_top
    )  # [J/s]
    expected_internal_energy = (
        3.0 * h_liq + 4.0 * h_vap - 101325.0 * 2.0 - 40.0
    )  # [J]
    np.testing.assert_allclose(
        result,
        [expected_energy_rate, expected_internal_energy],
    )
