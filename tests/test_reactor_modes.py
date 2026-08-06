# -*- coding: utf-8 -*-

import copy
import importlib.util
import json

import numpy as np
import pytest


HAS_ASSIMULO = importlib.util.find_spec("assimulo") is not None

pytestmark = [
    pytest.mark.assimulo,
    pytest.mark.integration,
    pytest.mark.skipif(
        not HAS_ASSIMULO,
        reason="assimulo is not installed; solver-backed integration tests skipped",
    ),
]

if HAS_ASSIMULO:
    from PharmaPy.Kinetics import RxnKinetics
    from PharmaPy.Phases import LiquidPhase
    from PharmaPy.Reactors import CSTR, BatchReactor, PlugFlowReactor
    from PharmaPy.Reactors import SemibatchReactor
    from PharmaPy.Streams import LiquidStream
    from PharmaPy.Utilities import CoolingWater

    SENSITIVITY_REACTORS = [CSTR, SemibatchReactor]
else:
    SENSITIVITY_REACTORS = []


def _load_pfr_config(data_path):
    with open(data_path["integration"] / "pfr_test_constructor_kwargs.json") as f:
        config = json.load(f)

    config = copy.deepcopy(config)
    tau = config["inlet"].pop("tau")
    config["inlet"]["vol_flow"] = config["phase"]["vol"] / tau

    datapath = str(data_path["integration"] / "pfr_test_pure_comp.json")
    config["kinetics"].update({
        "stoich_matrix": [[-1, -1, 1], [0, -1, 1]],
        "k_params": [40 / 60, 10 / 60],
        "ea_params": [2e3, 1e3],
        "delta_hrxn": [-5e3, -2.5e3],
    })
    config["kinetics"]["path"] = datapath

    return config, datapath


def _reactor_objects(data_path, reactor):
    config, datapath = _load_pfr_config(data_path)

    inlet = LiquidStream(datapath, **config["inlet"])
    phase = LiquidPhase(datapath, **config["phase"])
    kinetics = RxnKinetics(**config["kinetics"])
    utility = CoolingWater(**config["utility"])

    reactor.Inlet = inlet
    reactor.Phases = phase
    reactor.Kinetics = kinetics
    reactor.Utility = utility

    return reactor


def test_pfr_solve_steady_reads_inlet_mole_conc(data_path):
    config, _ = _load_pfr_config(data_path)
    reactor = _reactor_objects(
        data_path, PlugFlowReactor(**config["reactor"])
    )

    vol_position, states = reactor.solve_steady(reactor.Liquid_1.vol)

    vol_position = np.asarray(vol_position)

    assert vol_position.size > 1
    assert states.shape[0] == vol_position.size
    assert reactor.concProfSteady.shape[0] == vol_position.size
    assert reactor.Kinetics.num_rxns == 2
    assert reactor.tempProfSteady[-1] > reactor.Inlet.temp

    # With the correct 4/D specific area the 1 inch tube is strongly coupled
    # to the utility, so the profile equilibrates to it rather than merely
    # staying below it: a bare `< temp_ht_steady` bound holds here by ~4e-12 K,
    # which is inside the solver tolerance and would pass for the wrong reason.
    assert reactor.tempProfSteady[-1] == pytest.approx(
        reactor.temp_ht_steady, abs=1e-6)
    assert reactor.tempProfSteady[-1] <= reactor.temp_ht_steady


def test_steady_pfr_specific_area_matches_tube_geometry(data_path):
    """Pin the steady-PFR specific heat-transfer area to 4/D (Refs #33).

    Probing at zero reactant concentration makes every reaction rate zero, so
    the source term vanishes exactly and the steady energy balance reduces to
    ``dT/dV = -u_ht * a_prime * (T - T_ht) / (vol_flow * cp_vol)``. That lets
    ``a_prime`` [m**2/m**3] be recovered from one call and compared against
    the tube geometry 4/D -- the same expression the transient balance uses.
    The reciprocal form D/4 fails this by a factor of 16/D**2.

    Zeroing ``delta_hrxn`` would not work here: ``getHeatOfRxn`` applies a
    heat-capacity correction between ``tref_hrxn`` and the probe temperature,
    so the heat of reaction is nonzero even when the reference value is zero.
    """
    config, datapath = _load_pfr_config(data_path)

    reactor = PlugFlowReactor(**config["reactor"])
    reactor.Inlet = LiquidStream(datapath, **config["inlet"])
    reactor.Phases = LiquidPhase(datapath, **config["phase"])
    reactor.Kinetics = RxnKinetics(**config["kinetics"])
    reactor.Utility = CoolingWater(**config["utility"])

    reactor.solve_steady(reactor.Liquid_1.vol)

    temp_probe = 310.0  # [K], held away from the utility temperature
    conc_probe = np.zeros_like(reactor.concProfSteady[0])  # [mol/L], no rates

    dtemp_dv = float(reactor.energy_steady(conc_probe, temp_probe))  # [K/m**3]

    concentr = np.zeros_like(reactor.Liquid_1.mole_conc)  # [mol/L]
    concentr[reactor.mask_species] = conc_probe
    concentr[~reactor.mask_species] = reactor.c_inert
    _, cp_j = reactor.Liquid_1.getCpPure(temp_probe)  # [J/mol/K]
    cp_vol = np.dot(cp_j, concentr) * 1000  # [J/m**3/K]
    flow_term = reactor.Inlet.vol_flow * cp_vol  # [W/K]

    a_prime = -dtemp_dv * flow_term / (
        reactor.u_ht * (temp_probe - reactor.temp_ht_steady))  # [m**2/m**3]

    assert a_prime == pytest.approx(4 / reactor.diam, rel=1e-8)
    # Hand-computed for the fixture's 0.0254 m (1 inch) tube: 4/0.0254.
    assert a_prime == pytest.approx(157.4803, rel=1e-4)


@pytest.mark.parametrize("reactor_cls", SENSITIVITY_REACTORS)
def test_sensitivity_mode_refuses_unsupported_reactors(data_path, reactor_cls):
    if reactor_cls is CSTR:
        reactor = reactor_cls()
    else:
        reactor = reactor_cls(vol_tank=0.002)

    reactor = _reactor_objects(data_path, reactor)

    with pytest.raises(NotImplementedError, match="sensitivity.*not supported"):
        reactor.solve_unit(runtime=1, eval_sens=True, verbose=False)


def test_coil_ht_mode_refuses_unsupported_heat_transfer():
    reactor = BatchReactor(isothermal=False, ht_mode="coil")

    with pytest.raises(NotImplementedError, match="coil.*not supported"):
        reactor.heat_transfer(np.array([300.0]), np.array([290.0]), 0.002)


@pytest.mark.parametrize("reactor_cls", SENSITIVITY_REACTORS)
def test_coil_ht_mode_refuses_through_solve_unit(data_path, reactor_cls):
    if reactor_cls is CSTR:
        reactor = reactor_cls(isothermal=False, ht_mode="coil")
    else:
        reactor = reactor_cls(
            vol_tank=0.002, isothermal=False, ht_mode="coil")

    reactor = _reactor_objects(data_path, reactor)

    with pytest.raises(NotImplementedError, match="coil.*not supported"):
        reactor.solve_unit(runtime=1, verbose=False)
