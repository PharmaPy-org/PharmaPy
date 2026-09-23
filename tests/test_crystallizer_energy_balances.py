"""Test crystallizer energy balance regressions.

This module covers unit-basis and utility-jacket paths for crystallizer energy
balances.
"""

import sys
import types

import numpy as np
import pytest

_MISSING = object()
_ASSIMULO_STUBS = {}

try:
    from assimulo.problem import Explicit_Problem  # noqa: F401
    from assimulo.solvers import CVode  # noqa: F401
except ImportError:
    for module_name in ("assimulo", "assimulo.problem", "assimulo.solvers"):
        _ASSIMULO_STUBS[module_name] = sys.modules.get(module_name, _MISSING)
        sys.modules[module_name] = types.ModuleType(module_name)

    sys.modules["assimulo"].problem = sys.modules["assimulo.problem"]
    sys.modules["assimulo"].solvers = sys.modules["assimulo.solvers"]
    sys.modules["assimulo.problem"].Explicit_Problem = object
    sys.modules["assimulo.solvers"].CVode = object

try:
    from PharmaPy.Crystallizers import MSMPR, SemibatchCryst
finally:
    for module_name, previous in reversed(_ASSIMULO_STUBS.items()):
        if previous is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous

from PharmaPy.MixedPhases import Slurry


class StubPhase:
    kv = 0.5  # [-]

    def updatePhase(self, **kwargs):
        pass

    def getCp(self, temp=None, basis="mass"):
        cp_mass = 4000.0  # [J/kg/K]
        return cp_mass

    def getDensity(self, temp=None):
        density = 1000.0  # [kg/m**3]
        return density

    def getEnthalpy(self, temp=None, basis="mass"):
        h_mass = 1.0e5  # [J/kg]
        return h_mass


class StubSlurry:
    vol = 1.0e-3  # [m**3]
    temp_ht = None

    def getDensity(self, temp=None):
        density = np.array([1000.0, 2000.0])  # [kg/m**3]
        return density

    def getEnthalpy(self, temp, volfracs, density):
        h_vol = 2.0e8  # [J/m**3]
        return h_vol

    def getCp(self, temp, volfracs, density, times_vliq=False):
        cp_vol = 3.0e6  # [J/m**3/K]
        return cp_vol


class StubUtility:
    cp = 3500.0  # [J/kg/K]
    rho = 800.0  # [kg/m**3]

    def get_inputs(self, time):
        temp_in = 285.0  # [K]
        vol_flow = 2.0e-5  # [m**3/s]
        return {"temp_in": temp_in, "vol_flow": vol_flow}


class LiquidInlet:
    def getDensity(self, temp=None):
        rho_liq = 950.0  # [kg/m**3]
        return rho_liq

    def getEnthalpy(self, temp=None):
        h_mass = 123.0  # [J/kg]
        return h_mass


def _common_energy_attrs(cryst):
    """Supply the model and utility attributes normally set by construction.

    Parameters
    ----------
    cryst : MSMPR or SemibatchCryst
        Bare model instance receiving synthetic energy-balance collaborators.
    """
    cryst.Solid_1 = StubPhase()
    cryst.Slurry = StubSlurry()
    cryst.controls = {}

    cryst.diam_tank = 0.1  # [m]
    cryst.area_base = 0.01  # [m**2]
    cryst.u_ht = 500.0  # [J/s/m**2/K]
    cryst.vol_tank = 1.0e-3  # [m**3]
    cryst.vol_offset = 0.75  # [-], established working/vessel volume design ratio


COMMON_ENERGY_KW = {
    "time": 0.0,  # [s]
    "params": {},
    "cryst_rate": 0.0,  # [kg/s]
    "u_inputs": {"Inlet": {"vol_flow": 1.0e-6}},  # [m**3/s]
    "rhos": [np.array([1000.0, 2000.0]), np.array([1000.0, None])],  # [kg/m**3]
    "distrib": None,
    "mass_conc": None,
    "temp": 300.0,  # [K]
    "vol": 1.0e-3,  # [m**3]
    "h_in": 1.0e5,  # [J/m**3]
}


def test_msmpr_adiabatic_energy_balance_has_no_jacket_equation():
    cryst = MSMPR.__new__(MSMPR)
    _common_energy_attrs(cryst)
    cryst.adiabatic = True
    cryst.states_uo = ["distrib", "mass_conc", "vol", "temp"]

    dtemp_dt = cryst.energy_balances(  # [K/s]
        mu_n=np.zeros(4), temp_ht=None, **COMMON_ENERGY_KW
    )

    assert np.ndim(dtemp_dt) == 0


def test_semibatch_jacket_uses_utility_inputs():
    cryst = SemibatchCryst.__new__(SemibatchCryst)
    _common_energy_attrs(cryst)
    cryst.adiabatic = False
    cryst._Utility = StubUtility()

    temp_ht = 290.0  # [K]
    dtemp_dt, dtht_dt = cryst.energy_balances(  # [K/s]
        mu_n=np.zeros(4), temp_ht=temp_ht, **COMMON_ENERGY_KW
    )

    assert np.isfinite(dtemp_dt)
    assert np.isfinite(dtht_dt)


def test_liquid_feed_enthalpy_passed_to_energy_balance_is_volumetric():
    cryst = MSMPR.__new__(MSMPR)
    cryst.dim_states = [4, 1, 1]
    cryst.name_states = ["mu_n", "mass_conc", "temp"]
    cryst.method = "moments"
    cryst.states_di = {"mu_n": {"dim": 4}}
    cryst.scale = 1.0  # [-]
    cryst.controls = {}
    cryst.Slurry = StubSlurry()
    cryst.Liquid_1 = StubPhase()
    cryst.Solid_1 = StubPhase()
    cryst._Inlet = LiquidInlet()

    inlet_temp = 305.0  # [K]
    inlet_vol_flow = 2.0e-6  # [m**3/s]
    cryst.get_inputs = lambda time: {
        "Inlet": {"temp": inlet_temp, "vol_flow": inlet_vol_flow}
    }

    captured = {}

    def material_balances(*args, **kwargs):
        material_rates = np.zeros(5)  # [kg/m**3/s]
        cryst_rate = 0.0  # [kg/s]
        return material_rates, cryst_rate

    def energy_balances(*args, h_in, **kwargs):
        captured["h_in"] = h_in
        return h_in

    cryst.material_balances = material_balances
    cryst.energy_balances = energy_balances

    moments = np.zeros(4)  # [m**n/m**3]
    mass_conc = np.array([10.0])  # [kg/m**3]
    temp = np.array([300.0])  # [K]
    states = np.concatenate([moments, mass_conc, temp])

    result = cryst.unit_model(0.0, states, params={}, enrgy_bce=True)  # [J/m**3]

    expected_h_in = 123.0 * 950.0  # [J/kg]*[kg/m**3] -> [J/m**3]
    assert result == pytest.approx(expected_h_in)
    assert captured["h_in"] == pytest.approx(expected_h_in)


def test_slurry_getcp_times_vliq_does_not_mutate_volfracs():
    slurry = Slurry.__new__(Slurry)
    slurry.Liquid_1 = StubPhase()
    slurry.Solid_1 = StubPhase()

    volfracs = [0.9, 0.1]  # [-]
    density = np.array([1000.0, 2000.0])  # [kg/m**3]
    slurry.getCp(300.0, volfracs, density, times_vliq=True)

    assert volfracs == pytest.approx([0.9, 0.1])
