"""Scope of the overconsumption clamp in RxnKinetics.get_rxn_rates."""

import os

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics


pytestmark = pytest.mark.unit


DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

REACTIONS = ["A + B --> C"]
ACTIVATION_ENERGY = 0.0  # [J/mol], so the Arrhenius factor is exactly unity
# Large enough that one second of the unclamped rate would consume far more of
# the limiting reactant than is present.
RATE_CONSTANT = 1.0e4  # [L/mol/s]

# [mol/L], ordered as the participating species A, B, C. A is scarce, so the
# unclamped extent overshoots its inventory.
CONC = np.array([1.0e-3, 1.0, 0.0])
TEMP = 298.15  # [K]


def _kinetics():
    """Build a single irreversible reaction with a very large rate constant.

    Returns
    -------
    RxnKinetics
        Kinetics for ``A + B --> C`` whose unclamped rate at CONC would drive
        species A negative within one second.
    """
    return RxnKinetics(
        path=DATA_PATH,
        rxn_list=REACTIONS,
        k_params=np.array([RATE_CONSTANT]),
        ea_params=np.array([ACTIVATION_ENERGY]),
    )


def _unclamped_species_rates(kinetics, conc, temp):
    """Species rates implied by the per-reaction rates, with no clamping.

    Per-reaction rates are returned raw, so projecting them through the
    stoichiometry reproduces what the species rates would be if nothing
    rescaled them.

    Parameters
    ----------
    kinetics : RxnKinetics
        Kinetics object under test.
    conc : ndarray
        Participating species concentrations [mol/L].
    temp : float
        Temperature [K].

    Returns
    -------
    ndarray
        Species rates [mol/L/s].
    """
    per_reaction = kinetics.get_rxn_rates(conc, temp, overall_rates=False)
    return np.dot(per_reaction, kinetics.normalized_stoich.T)


def test_public_species_rates_are_not_clamped():
    """The default call reports the rate the rate law actually gives.

    The analytic concentration Jacobian is differentiated from the unclamped
    rate law and knows nothing about the rescaling, so a clamped public rate
    would hand a solver a rate and a Jacobian that disagree. Overconsumption
    is a question for the balance that integrates these rates, not for the
    rate law that reports them.
    """
    kinetics = _kinetics()

    rates = kinetics.get_rxn_rates(CONC, TEMP)

    np.testing.assert_allclose(
        rates, _unclamped_species_rates(kinetics, CONC, TEMP), rtol=1e-12
    )


def test_the_mechanism_call_still_clamps_to_the_inventory():
    """The refactored mechanisms ask for clamped rates explicitly.

    PharmaPy.Mechanisms is the only caller that passes return_both, and it
    relies on the rescaling so a pseudo-instantaneous reaction cannot consume
    more of a species than the phase holds.
    """
    kinetics = _kinetics()

    _, species_rates = kinetics.get_rxn_rates(CONC, TEMP, return_both=True)

    limiting = 0  # index of A within the participating species [-]
    assert species_rates[limiting] > _unclamped_species_rates(
        kinetics, CONC, TEMP
    )[limiting]
    # One second at the clamped rate must not drive the limiting species below
    # zero, which is the whole point of the rescaling.
    assert CONC[limiting] + species_rates[limiting] >= -np.finfo(float).eps
