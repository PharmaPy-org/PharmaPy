"""Adiabatic drum energy includes the enthalpy carried by reflux condensate."""

import numpy as np
import pytest

from PharmaPy.Evaporators import ContinuousEvaporator


pytestmark = pytest.mark.unit

BALANCE_RTOL = 1e-12  # [-], roundoff allowance for the short enthalpy sum


class _EnthalpySource:
    """Provide fixed enthalpy [J/mol] and bubble point [K] test values."""

    def __init__(self, enthalpy):
        self.enthalpy = enthalpy  # [J/mol]

    def getEnthalpy(self, *args, **kwargs):
        return self.enthalpy  # [J/mol]

    def getBubblePoint(self, *args, **kwargs):
        return 325.0  # [K]


@pytest.mark.parametrize(
    "reflux_ratio, expected_energy_rate",
    [
        (0, -40.0),  # [-], [J/s]
        (0.25, -30.0),  # [-], [J/s]
    ],
)
def test_adiabatic_energy_residual_includes_vapor_enthalpy(
        reflux_ratio, expected_energy_rate):
    """Check F*h_F - L*h_L - V*h_V + r*V*h_top for an adiabatic drum.

    The fixture gives 4*10 - 1*20 - 2*30 + r*2*20 [J/s], hence -40 J/s
    without reflux and -30 J/s at r=0.25. The external condenser removes
    2*(30-20)=20 J/s when reflux is enabled; adiabatic means no jacket duty.

    Parameters
    ----------
    reflux_ratio : float
        Fraction of vapor returned as condensate [-].
    expected_energy_rate : float
        Independently derived drum energy accumulation [J/s].
    """
    # Enthalpies are J/mol; flows are mol/s, amounts are mol, pressure is Pa,
    # and volume is m^3. Thus the two residuals are J/s and J, respectively.
    evaporator = ContinuousEvaporator.__new__(ContinuousEvaporator)
    evaporator._Inlet = _EnthalpySource(10.0)  # [J/mol]
    evaporator.Liquid_1 = _EnthalpySource(20.0)  # [J/mol]
    evaporator.Vapor_1 = _EnthalpySource(30.0)  # [J/mol]
    evaporator.reflux_ratio = reflux_ratio  # [-]
    evaporator.activity_model = 'ideal'
    evaporator.adiabatic = True  # [-]
    evaporator.vol_tot = 2.0  # [m^3]

    result = evaporator.energy_balances(  # [J/s, J]
        time=0.0,
        flow_liq=1.0,
        flow_vap=2.0,
        vol_liq=1.0,
        u_int=40.0,
        temp=350.0,
        x_liq=np.array([1.0]),
        y_vap=np.array([1.0]),
        mol_i=np.array([3.0]),
        mol_liq=3.0,
        mol_vap=4.0,
        pres=5.0,
        u_inputs={
            "mole_flow": 4.0,
            "mole_frac": np.array([1.0]),
            "temp": 300.0,
        },
    )

    expected_internal_energy = 3.0 * 20.0 + 4.0 * 30.0 - \
        5.0 * 2.0 - 40.0  # [J]
    np.testing.assert_allclose(
        result,
        [expected_energy_rate, expected_internal_energy],
        rtol=BALANCE_RTOL,
    )
