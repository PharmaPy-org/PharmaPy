"""Algebra-only reactor energy-balance regressions without Assimulo."""

import numpy as np
import pytest

import PharmaPy.Reactors as reactors
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


HOLDUP_CONCENTRATION = np.array([0.0, 1.0, 0.0, 0.0])  # [mol/L]
INLET_CONCENTRATION = np.array([1.0, 0.0, 0.0, 0.0])  # [mol/L]
TEMPERATURE = 350.0  # [K]
VOLUMETRIC_FLOW = 2.0  # [m**3/s]


def _configure_energy_balance(reactor, data_path):
    """Attach real thermophysical, stream, and kinetics collaborators.

    Parameters
    ----------
    reactor : CSTR or SemibatchReactor
        Production reactor receiving the real collaborators.
    data_path : dict
        Repository test-data paths.

    Returns
    -------
    LiquidPhase
        Holdup phase used to derive the independent enthalpy expectation.
    """
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    liquid = LiquidPhase(
        thermo_path,
        temp=TEMPERATURE,
        vol=1.0,  # [m**3]
        mole_conc=HOLDUP_CONCENTRATION,
        verbose=False,
    )
    inlet = LiquidStream(
        thermo_path,
        temp=TEMPERATURE,
        vol_flow=VOLUMETRIC_FLOW,
        mole_conc=INLET_CONCENTRATION,
        verbose=False,
    )
    kinetics = RxnKinetics(
        thermo_path,
        k_params=[0.0],  # [1/s]
        ea_params=[0.0],  # [J/mol]
        stoich_matrix=np.array([[-1.0, 1.0, 0.0, 0.0]]),  # [-]
        partic_species=liquid.name_species,
        delta_hrxn=[0.0],  # [J/mol_rxn]
    )

    reactor.Phases = liquid
    reactor.Kinetics = kinetics
    reactor.Inlet = inlet
    reactor.set_names()

    return liquid


def _flow_term(reactor):
    inputs = {
        "Inlet": {
            "vol_flow": VOLUMETRIC_FLOW,
            "mole_conc": INLET_CONCENTRATION,
            "temp": TEMPERATURE,
        }
    }

    heat_profile = reactor.energy_balances(
        0.0,
        HOLDUP_CONCENTRATION,
        1.0,
        TEMPERATURE,
        TEMPERATURE,
        inputs,
        heat_prof=True,
    )

    return float(heat_profile[0, -1])


def test_semibatch_flow_term_uses_inlet_composition_for_sensible_enthalpy(
        data_path):
    reactor = reactors.SemibatchReactor(vol_tank=1.0, isothermal=True)
    _configure_energy_balance(reactor, data_path)

    assert _flow_term(reactor) == pytest.approx(0.0)


def test_cstr_flow_term_keeps_holdup_composition_for_outflow(data_path):
    reactor = reactors.CSTR(isothermal=True)
    liquid = _configure_energy_balance(reactor, data_path)

    species_enthalpy = np.ravel(liquid.getEnthalpy(
        TEMPERATURE,
        reactor.temp_ref,
        total_h=False,
        basis="mole",
    ))  # [J/mol]
    expected_flow_term = VOLUMETRIC_FLOW * np.dot(
        INLET_CONCENTRATION - HOLDUP_CONCENTRATION,
        species_enthalpy,
    ) * 1000  # [W]

    assert _flow_term(reactor) == pytest.approx(expected_flow_term)
