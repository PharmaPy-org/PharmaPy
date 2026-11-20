"""Crystals fed between refactored crystallizers in each population form.

A continuous crystallizer (U01) feeds an unseeded semibatch one (U02). Every
rate law is off, so seeds only move: what leaves U01 must arrive in U02, or
the feed must be refused when the two units carry the crystals in different
forms.

Units: x_grid [um], distrib [#/(um m**3)], moments [um**k/m**3], both per
m**3 of slurry; feed [kg/s]; time [s].
"""

import os

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import (ContinuousCrystallizer,
                                               SemiBatchCrystallizer)
from PharmaPy.Errors import PharmaPySpecificationError
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import MomentsPopulationBalance, OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams_Refactored import LiquidStream

pytestmark = pytest.mark.integration

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

LIQUID_MASS_FRAC = [0.0, 0.0, 0.2, 0.0, 0.8]  # [-], A, B, C, D, solvent
CRYSTAL_MASS_FRAC = [0.0, 0.0, 1.0, 0.0, 0.0]  # [-], pure C
TEMP = 298.15  # [K], isothermal
CHARGE_MASS = 1.0  # [kg] of liquid in each unit
FEED_MASS_FLOW = 1e-3  # [kg/s] of liquid into U01
RUNTIME = 600.0  # [s]

# Every rate constant is zero, so the solubility only has to be valid.
SOLUBILITY_COEFFS = [50.0]  # [kg/m**3], constant
NO_PRIMARY_NUCLEATION = (0.0, 0.0, 1.0)  # k [#/(m**3 s)], E [J/mol], n [-]
NO_SECONDARY_NUCLEATION = (0.0, 0.0, 1.0, 1.0)  # k, E [J/mol], n, s_2 [-]
NO_GROWTH = (0.0, 0.0, 1.0)  # k [um/s], E [J/mol], n [-]
NO_DISSOLUTION = (0.0, 0.0, 1.0)  # k [um/s], E [J/mol], n [-]

CRYSTAL_DENSITY = 1200.0  # [kg/m**3]
SHAPE_FACTOR = 1.0  # [-]
GRID_STEP = 1.0  # [um], uniform, so every cell is this wide
SIZE_GRID = np.arange(1.0, 200.0, GRID_STEP)  # [um]
SEED_SIZE = 50.0  # [um]
SEED_WIDTH = 5.0  # [um], standard deviation of the resolved seed peak
SEED_COUNT = 1e9  # [#/m**3 of slurry]

# LSODA at rtol = atol = 1e-8. Against populations of order 1e8 to 1e14 the
# absolute tolerance never binds, so this is relative error control.
SOLVER_TOLERANCE = 1e-8  # [-]
# Crystals arriving in U02 matched those leaving U01 to 1.2e-7 for moments
# and 1.8e-9 for the distribution at that solver tolerance; 1e-6 keeps an
# eight-fold margin and still fails on any lost fraction above it.
FLOW_SOLVE_RTOL = 1e-6  # [-]
# About 45 % of the seeds leave U01 within RUNTIME, so a run that moves
# fewer than a quarter of them cannot test the handoff.
MIN_FRACTION_LEFT = 0.25  # [-]

# A refusal names the feed's population state first, then the vessel's.
DISTRIBUTION_INTO_MOMENTS = (r"OneDFVMMechanism state 'distrib' .* "
                             r"MomentsPopulationBalance carries them as 'mu_n'")
MOMENTS_INTO_DISTRIBUTION = (r"MomentsPopulationBalance state 'mu_n' .* "
                             r"OneDFVMMechanism carries them as 'distrib'")


def _solid(form, seeded):
    """Crystal phase carrying its seeds as a distribution or as moments.

    Parameters
    ----------
    form : {'moments', 'fvm'}
        Population form of the crystals.
    seeded : bool
        Whether the phase starts with ``SEED_COUNT`` seeds per m**3 of
        slurry, or with none.

    Returns
    -------
    SolidPhase
        Solid phase with its population mechanism attached.
    """
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC,
                       temp=TEMP)
    count = SEED_COUNT if seeded else 0.0  # [#/m**3 of slurry]

    if form == "moments":
        # Seeds of one size: mu_k = N L**k.
        moments = count * SEED_SIZE ** np.arange(4)  # [um**k/m**3]
        solid.mechanisms = MomentsPopulationBalance(
            solid, target_components="C", solvent_name="solvent",
            moments_init=moments, density=CRYSTAL_DENSITY, kv=SHAPE_FACTOR)
    else:
        peak = np.exp(-0.5 * ((SIZE_GRID - SEED_SIZE) / SEED_WIDTH) ** 2)  # [-]
        distrib = (count / (SEED_WIDTH * np.sqrt(2 * np.pi))
                   * peak)  # [#/(um m**3)]
        solid.mechanisms = OneDFVMMechanism(
            solid, target_components="C", solvent_name="solvent",
            x_grid=SIZE_GRID, distrib_init=distrib, density=CRYSTAL_DENSITY,
            kv=SHAPE_FACTOR)

    return solid


def _unit(vessel_cls, form, seeded):
    """Liquid-fed crystallizer in which nothing forms, grows or dissolves.

    Parameters
    ----------
    vessel_cls : type
        ``ContinuousCrystallizer`` or ``SemiBatchCrystallizer``.
    form : {'moments', 'fvm'}
        Population form of its crystals.
    seeded : bool
        Whether it starts with seeds.

    Returns
    -------
    ContinuousCrystallizer or SemiBatchCrystallizer
        Unit ready for a flowsheet.
    """
    integrator = ScipyBackend(method="LSODA",
                              options={"rtol": SOLVER_TOLERANCE,
                                       "atol": SOLVER_TOLERANCE})
    vessel = vessel_cls(integrator=integrator, isothermal=True)
    vessel.Phases = [LiquidPhase(DATA_PATH, mass=CHARGE_MASS,
                                 mass_frac=LIQUID_MASS_FRAC, temp=TEMP,
                                 name_solv="solvent"),
                     _solid(form, seeded)]
    vessel.CrystKinetics = CrystKinetics(
        coeff_solub=SOLUBILITY_COEFFS, nucl_prim=NO_PRIMARY_NUCLEATION,
        nucl_sec=NO_SECONDARY_NUCLEATION, growth=NO_GROWTH,
        dissolution=NO_DISSOLUTION)
    vessel.Inlet = LiquidStream(DATA_PATH, mass_flow=FEED_MASS_FLOW,
                                mass_frac=LIQUID_MASS_FRAC, temp=TEMP,
                                name_solv="solvent")
    return vessel


def _solve(upstream_form, downstream_form, seeded=True):
    """Run U01 (continuous) into U02 (unseeded semibatch).

    Parameters
    ----------
    upstream_form, downstream_form : {'moments', 'fvm'}
        Population forms of U01 and U02.
    seeded : bool, optional
        Whether U01 starts with seeds.

    Returns
    -------
    SimulationExec
        The solved flowsheet.
    """
    flowsheet = SimulationExec(DATA_PATH, flowsheet="U01 --> U02")
    flowsheet.U01 = _unit(ContinuousCrystallizer, upstream_form, seeded)
    flowsheet.U02 = _unit(SemiBatchCrystallizer, downstream_form, False)
    run = {"runtime": RUNTIME, "verbose": False}
    flowsheet.SolveFlowsheet(kwargs_run={"U01": run, "U02": run},
                             verbose=False)
    return flowsheet


def _crystal_count(vessel, form):
    """Crystals held in the vessel over its solve.

    Parameters
    ----------
    vessel : ContinuousCrystallizer or SemiBatchCrystallizer
        Solved unit.
    form : {'moments', 'fvm'}
        Population form of its crystals.

    Returns
    -------
    numpy.ndarray
        Crystal count [#] at each output time, shape ``(num_times,)``.
    """
    result = vessel.result
    volume = np.ravel(result.vessel_vol)  # [m**3] of slurry

    if form == "moments":
        number_density = np.asarray(result.mu_n_solid0)[:, 0]  # [#/m**3]
    else:
        number_density = (np.sum(np.asarray(result.distrib_solid0), axis=1)
                          * GRID_STEP)  # [#/m**3]

    return number_density * volume


@pytest.mark.parametrize("form", ["moments", "fvm"])
def test_crystals_fed_in_the_vessel_form_arrive(form):
    flowsheet = _solve(form, form)

    held = _crystal_count(flowsheet.U01, form)
    left = held[0] - held[-1]  # [#]

    assert left > MIN_FRACTION_LEFT * held[0]
    assert _crystal_count(flowsheet.U02, form)[-1] == pytest.approx(
        left, rel=FLOW_SOLVE_RTOL)


@pytest.mark.parametrize("feed_form, vessel_form, message", [
    ("fvm", "moments", DISTRIBUTION_INTO_MOMENTS),
    ("moments", "fvm", MOMENTS_INTO_DISTRIBUTION),
], ids=["distribution_into_moments", "moments_into_distribution"])
def test_a_feed_in_another_population_form_is_refused(feed_form, vessel_form,
                                                      message):
    with pytest.raises(PharmaPySpecificationError, match=message):
        _solve(feed_form, vessel_form)


def test_an_empty_feed_in_another_form_is_refused_too():
    # The forms are what cannot be joined, so the refusal must not wait for
    # the first crystal to arrive part way through the solve.
    with pytest.raises(PharmaPySpecificationError,
                       match=DISTRIBUTION_INTO_MOMENTS):
        _solve("fvm", "moments", seeded=False)
