import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_INIT = REPO_ROOT / "PharmaPy" / "__init__.py"


def _stub_assimulo_modules(monkeypatch):
    """Let algebra-only reactor tests import without the optional solver."""
    assimulo = ModuleType("assimulo")

    solvers = ModuleType("assimulo.solvers")
    solvers.CVode = object
    solvers.LSODAR = object

    problem = ModuleType("assimulo.problem")
    problem.Explicit_Problem = object

    monkeypatch.setitem(sys.modules, "assimulo", assimulo)
    monkeypatch.setitem(sys.modules, "assimulo.solvers", solvers)
    monkeypatch.setitem(sys.modules, "assimulo.problem", problem)


def _prefer_source_package():
    """Avoid importing the outer checkout package named PharmaPy."""
    loaded = sys.modules.get("PharmaPy")
    loaded_path = getattr(loaded, "__file__", None)
    if loaded is not None and (
            loaded_path is None or Path(loaded_path).resolve() != PACKAGE_INIT):
        del sys.modules["PharmaPy"]

    try:
        sys.path.remove(str(REPO_ROOT))
    except ValueError:
        pass
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def reactors(monkeypatch):
    _prefer_source_package()

    try:
        import PharmaPy.Reactors as reactors_module
    except ModuleNotFoundError as exc:
        if exc.name != "assimulo":
            raise
        _stub_assimulo_modules(monkeypatch)

        import PharmaPy.Reactors as reactors_module

    return reactors_module


class _StubLiquid:
    def __init__(self, heat_capacities):
        self.heat_capacities = np.asarray(heat_capacities, dtype=float)

    def getCpPure(self, temp):
        return None, self.heat_capacities

    def getEnthalpy(self, temp, temp_ref, total_h=False, basis="mole"):
        temp = np.atleast_1d(temp)
        return self.heat_capacities[None, :] * (temp[:, None] - temp_ref)

    def getHeatOfRxn(self, stoich, temp, mask, dh_ref, tref):
        return np.zeros((1, 1))


class _StubKinetics:
    delta_hrxn = np.zeros(1)
    stoich_matrix = np.zeros((1, 2))
    stoich_normalization = np.ones(1)  # [-]
    tref_hrxn = 298.15

    def get_rxn_rates(self, conc, temp, overall_rates=False, delta_hrxn=None):
        return np.zeros((1, 1))


def _configure_energy_balance_stubs(reactor):
    liquid = _StubLiquid([100.0, 200.0])

    reactor.mask_species = np.array([True, True])
    reactor.Liquid_1 = liquid
    reactor.Inlet = liquid
    reactor._Kinetics = _StubKinetics()


def _flow_term(reactor):
    inputs = {
        "Inlet": {
            "vol_flow": 2.0,
            "mole_conc": np.array([1.0, 0.0]),
            "temp": 350.0,
        }
    }

    heat_profile = reactor.energy_balances(
        0.0,
        np.array([0.0, 1.0]),
        1.0,
        350.0,
        350.0,
        inputs,
        heat_prof=True,
    )

    return float(heat_profile[0, -1])


def test_semibatch_flow_term_uses_inlet_composition_for_sensible_enthalpy(
        reactors):
    reactor = reactors.SemibatchReactor(vol_tank=1.0, isothermal=True)
    _configure_energy_balance_stubs(reactor)

    assert _flow_term(reactor) == pytest.approx(0.0)


def test_cstr_flow_term_keeps_holdup_composition_for_outflow(reactors):
    reactor = reactors.CSTR(isothermal=True)
    _configure_energy_balance_stubs(reactor)

    assert _flow_term(reactor) == pytest.approx(-10370000.0)
