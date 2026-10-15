"""The batch crystallizer solved on CVode with the flowsheet case-study kinetics.

Fixture: the cooling crystallizer of ``tests/Flowsheet``, charged with the
reactor product and cooled from 313.15 to 278.15 K over 2 h on 35 geometric
size classes. Nucleation starts about 20 min in, which is where CVode used to
stop. An independent integrator, SciPy LSODA, is the reference, on this feed
and on two with less product C. One test solves the moment form instead, and
one checks the step cap that lets either linear solver resolve the burst.

Units: temp [K], conc [mol/L], vol [m**3], time [s], x_grid [um],
crystal count [#/m**3 of slurry], L4,3 [um].
"""

import importlib.util
import os

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.IntegratorBackends import AssimuloBackend, ScipyBackend
from PharmaPy.Interpolation import PiecewiseLagrange
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import MomentsPopulationBalance, OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl_Refactored import SimpleTemperatureController
from PharmaPy.Utilities import CoolingWater

pytestmark = [
    pytest.mark.assimulo,
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("assimulo") is None,
        reason="assimulo is not installed; CVode tests skipped",
    ),
]

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

# Case-study crystallization kinetics, as in tests/Flowsheet.
SOLUBILITY = np.array([2.269e2, -1.88e0, 3.89e-3])  # c_sat = a + b*T + c*T**2 [kg/m**3], T [K]
NUCL_PRIM = (3e8, 0, 3)  # k_p [#/m**3/s], E [J/mol], order [-]
NUCL_SEC = (4.46e10, 0, 2, 1e-5)  # k_s [#/m**3/s], E [J/mol], orders [-]
GROWTH = (5, 0, 1.32)  # k_g [um/s], E [J/mol], order [-]
DISSOLUTION = (1, 0, 1)  # k_d [um/s], E [J/mol], order [-]

# R01's product after 1 h at 313.15 K in the case study: A, B, C, D, solvent.
FEED_CONC = np.array([0.00231, 0.1036, 0.1250, 0.1013, 0.0])  # [mol/L]
PRODUCT_INDEX = 2  # C in FEED_CONC
FEED_TEMP = 313.15  # [K]
FEED_VOL = 0.06  # [m**3]
CRYSTAL_MASS_FRAC = [0.0, 0.0, 1.0, 0.0, 0.0]  # [-], pure C

SIZE_GRID = np.geomspace(1, 1500, num=35)  # [um]
# mu_0..mu_3: the third moment carries secondary nucleation and crystal mass.
NUM_MOMENTS = 4
COOLING_PROGRAM = np.array([[313.15, 308.0], [308.0, 295.0], [295.0, 278.15]])  # [K]
RUNTIME = 7200.0  # [s]
# The step cap the flowsheet tests used before 10 s. It lets CVode cross
# nucleation onset in one step, which is what exposed the defects most tests
# here guard.
MAX_STEP = 60.0  # [s]
# The cap the flowsheet tests use now: short enough for either linear solver
# to resolve the burst after nucleation onset.
ONSET_STEP_CAP = 10.0  # [s]

HEAT_TRANSFER_COEFF = 1e4  # [W/m**2/K], unused: the controller owns temperature
VESSEL_DIAMETER = 0.4  # [m]
UTILITY_MASS_FLOW = 1.0  # [kg/s]
UTILITY_TEMP_IN = 283.15  # [K]

# A nucleated batch here holds about 5e10 crystals per m**3; an empty or
# barely started one stays decades below this.
NUCLEATED_COUNT_FLOOR = 1e9  # [#/m**3]
# CVode's default absolute tolerance, and an explicit one distinct from it,
# both in each state's own units.
CVODE_DEFAULT_ATOL = 1e-6
EXPLICIT_ATOL = 1e-8
# The reference runs a hundred times tighter than CVode's default.
REFERENCE_RTOL = 1e-8  # [-]
# Product C concentrations the agreement test runs besides the case-study
# feed, other species unchanged, so at lower supersaturation. A design
# choice: on these two, at the MAX_STEP cap, CVode with the Krylov linear
# solver overcounted the crystals by 50 and 61 %, while on the case-study
# feed it happened to agree.
LEANER_PRODUCT_CONC = (0.095, 0.115)  # [mol/L]
# Leaner feed on which the moment form never crystallized when its moments
# carried tolerances scaled to their final values.
LEAN_MOMENTS_PRODUCT_CONC = 0.095  # [mol/L]
# Leaner feed on which, at MAX_STEP, CVode missed the crystal count with
# both linear solvers: dense by 4.8 %, Krylov by 39 %.
STEP_CAP_PRODUCT_CONC = 0.090  # [mol/L]
# Dense CVode and LSODA agree to about 1e-5 on all three feeds. 1e-3 leaves
# headroom across platforms and stays far below the Krylov errors.
AGREEMENT_RTOL = 1e-3  # [-]


def _build(integrator, feed_conc=FEED_CONC, moments=False, linear_solver=None):
    """Set up the case-study crystallizer, ready to solve.

    Parameters
    ----------
    integrator : IntegratorBackend
        Fresh backend for this vessel.
    feed_conc : numpy.ndarray, optional
        Feed concentrations of A, B, C, D and solvent [mol/L], shape
        ``(5,)``. Defaults to ``FEED_CONC``.
    moments : bool, optional
        Carry the population as its first ``NUM_MOMENTS`` moments instead
        of the size distribution. Defaults to False.
    linear_solver : {"dense", "krylov"}, optional
        Linear solver to request instead of the crystallizer's own choice.
        None, the default, keeps the crystallizer's choice.

    Returns
    -------
    tuple
        The ``BatchCrystallizer`` and its population balance mechanism.
    """
    solid = SolidPhase(DATA_PATH, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC)
    if moments:
        mechanism = MomentsPopulationBalance(
            owning_phase=solid,
            target_components="C",
            solvent_name="solvent",
            moments_init=np.zeros(NUM_MOMENTS),
        )
    else:
        mechanism = OneDFVMMechanism(
            solid,
            target_components="C",
            solvent_name="solvent",
            x_grid=SIZE_GRID,
            distrib_init=np.zeros_like(SIZE_GRID),
        )
    solid.mechanisms = mechanism

    cooling = PiecewiseLagrange(RUNTIME, COOLING_PROGRAM).evaluate_poly
    crystallizer = BatchCrystallizer(
        integrator=integrator,
        h_conv=HEAT_TRANSFER_COEFF,
        diam=VESSEL_DIAMETER,
        controller=SimpleTemperatureController(temp_func=cooling),
    )
    crystallizer.Phases = [
        LiquidPhase(DATA_PATH, temp=FEED_TEMP,
                    mole_conc=np.array(feed_conc, dtype=float),
                    vol=FEED_VOL, name_solv="solvent"),
        solid,
    ]
    crystallizer.CrystKinetics = CrystKinetics(
        SOLUBILITY, nucl_prim=NUCL_PRIM, nucl_sec=NUCL_SEC, growth=GROWTH,
        dissolution=DISSOLUTION,
    )
    crystallizer.Utility = CoolingWater(
        mass_flow=UTILITY_MASS_FLOW, temp_in=UTILITY_TEMP_IN
    )
    if linear_solver is not None:
        crystallizer.configure_solver = (
            lambda: crystallizer.integrator.set_linear_solver(linear_solver)
        )
    return crystallizer, mechanism


def _solve(integrator, feed_conc=FEED_CONC, linear_solver=None):
    """Solve the case-study crystallizer and summarise its final crystals.

    Parameters
    ----------
    integrator : IntegratorBackend
        Fresh backend for this vessel.
    feed_conc : numpy.ndarray, optional
        Feed concentrations of A, B, C, D and solvent [mol/L], shape
        ``(5,)``. Defaults to ``FEED_CONC``.
    linear_solver : {"dense", "krylov"}, optional
        Linear solver to request instead of the crystallizer's own choice.

    Returns
    -------
    numpy.ndarray
        Crystal count [#/m**3 of slurry] and L4,3 [um], shape ``(2,)``.
    """
    crystallizer, mechanism = _build(integrator, feed_conc, linear_solver=linear_solver)
    time, _ = crystallizer.solve_unit(runtime=RUNTIME, verbose=False)
    assert time[-1] == pytest.approx(RUNTIME)

    density = np.asarray(crystallizer.result.distrib_solid0)[-1]  # [#/(um m**3)]
    count = mechanism.integrate_over_size(density)  # [#/m**3]
    third = mechanism.integrate_over_size(density * SIZE_GRID**3)  # [um**3/m**3]
    fourth = mechanism.integrate_over_size(density * SIZE_GRID**4)  # [um**4/m**3]
    return np.array([count, fourth / third])


def _solve_moments(integrator, feed_conc):
    """Solve the crystallizer in moment form and return its final moments.

    Parameters
    ----------
    integrator : IntegratorBackend
        Fresh backend for this vessel.
    feed_conc : numpy.ndarray
        Feed concentrations of A, B, C, D and solvent [mol/L], shape
        ``(5,)``.

    Returns
    -------
    numpy.ndarray
        Crystal count [#/m**3 of slurry] and third moment
        [um**3/m**3 of slurry], shape ``(2,)``.
    """
    crystallizer, _ = _build(integrator, feed_conc, moments=True)
    time, _ = crystallizer.solve_unit(runtime=RUNTIME, verbose=False)
    assert time[-1] == pytest.approx(RUNTIME)

    moments = np.asarray(crystallizer.result.mu_n_solid0)[-1]  # [um**k/m**3]
    return moments[[0, 3]]


def test_cvode_integrates_through_nucleation_onset():
    """With a 60 s step cap, CVode reaches the end of the batch."""
    count, _ = _solve(AssimuloBackend(options={"maxh": MAX_STEP}))

    assert count > NUCLEATED_COUNT_FLOOR


def _compiled_solver(backend):
    """Compile the case-study crystallizer on ``backend`` without solving.

    Parameters
    ----------
    backend : AssimuloBackend
        Fresh backend for this vessel.

    Returns
    -------
    tuple
        The crystallizer and the CVode solver the backend built for it.
    """
    crystallizer, _ = _build(backend)
    crystallizer.compile_structure()
    backend.compile_integrator(crystallizer, verbose=False)
    return crystallizer, backend._solver


def test_cvode_takes_the_declared_tolerances_unless_atol_is_set():
    """CVode gets the vessel's tolerance vector; an explicit atol wins."""
    crystallizer, solver = _compiled_solver(
        AssimuloBackend(options={"maxh": MAX_STEP})
    )

    np.testing.assert_array_equal(
        solver.atol, crystallizer.solver_absolute_tolerances(CVODE_DEFAULT_ATOL)
    )

    _, solver = _compiled_solver(
        AssimuloBackend(options={"maxh": MAX_STEP, "atol": EXPLICIT_ATOL})
    )

    np.testing.assert_array_equal(solver.atol, EXPLICIT_ATOL)


@pytest.mark.parametrize(
    "product_conc", (FEED_CONC[PRODUCT_INDEX],) + LEANER_PRODUCT_CONC
)
def test_cvode_matches_an_independent_integrator(product_conc):
    """Crystal count and L4,3 agree with LSODA on the same model."""
    feed_conc = FEED_CONC.copy()  # [mol/L]
    feed_conc[PRODUCT_INDEX] = product_conc
    cvode = _solve(AssimuloBackend(options={"maxh": MAX_STEP}), feed_conc)
    reference = _solve(ScipyBackend(
        method="LSODA", options={"maxh": MAX_STEP, "rtol": REFERENCE_RTOL}
    ), feed_conc)

    assert reference[0] > NUCLEATED_COUNT_FLOOR
    np.testing.assert_allclose(cvode, reference, rtol=AGREEMENT_RTOL)


def test_cvode_crystallizes_a_lean_feed_in_moment_form():
    """The moment form nucleates on a lean feed and agrees with LSODA.

    With tolerances scaled to the final moments, CVode let the third moment
    sit just below zero, where secondary nucleation is off, and the batch
    never crystallized.
    """
    feed_conc = FEED_CONC.copy()  # [mol/L]
    feed_conc[PRODUCT_INDEX] = LEAN_MOMENTS_PRODUCT_CONC
    cvode = _solve_moments(AssimuloBackend(options={"maxh": MAX_STEP}), feed_conc)
    reference = _solve_moments(ScipyBackend(
        method="LSODA", options={"maxh": MAX_STEP, "rtol": REFERENCE_RTOL}
    ), feed_conc)

    assert reference[0] > NUCLEATED_COUNT_FLOOR
    np.testing.assert_allclose(cvode, reference, rtol=AGREEMENT_RTOL)


@pytest.mark.parametrize("linear_solver", ["dense", "krylov"])
def test_step_cap_resolves_nucleation_for_either_linear_solver(linear_solver):
    """Capped at ONSET_STEP_CAP, either linear solver matches LSODA.

    At MAX_STEP both crossed nucleation onset in one step on this feed and
    miscounted the crystals.
    """
    feed_conc = FEED_CONC.copy()  # [mol/L]
    feed_conc[PRODUCT_INDEX] = STEP_CAP_PRODUCT_CONC
    cvode = _solve(AssimuloBackend(options={"maxh": ONSET_STEP_CAP}), feed_conc,
                   linear_solver=linear_solver)
    reference = _solve(ScipyBackend(
        method="LSODA", options={"maxh": MAX_STEP, "rtol": REFERENCE_RTOL}
    ), feed_conc)

    assert reference[0] > NUCLEATED_COUNT_FLOOR
    np.testing.assert_allclose(cvode, reference, rtol=AGREEMENT_RTOL)
