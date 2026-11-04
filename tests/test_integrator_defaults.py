"""Default integrator, merged backend options and the crystallizer step cap.

Most tests build backends and vessels without solving, so they need no
Assimulo; the ones that compile a CVode solver are marked ``assimulo``. The
fallback without Assimulo runs in a child process that blocks its import.
Fixtures use the five-species ``tests/Flowsheet/data`` database.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.IntegratorBackends import AssimuloBackend, ScipyBackend
from PharmaPy.Kinetics import CrystKinetics, RxnKinetics
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Reactors_Refactored import BatchReactor

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]

# The 1 s cap CVode has always had by default.
BACKEND_DEFAULT_CAP = 1.0  # [s]
# A cap set by the caller, different from both the default and the
# crystallizers' request so that each can be told apart.
CALLER_CAP = 2.0  # [s]
CALLER_RTOL = 1e-8  # [-]

CHARGE_MASS = 1.0  # [kg]
REACTOR_MASS_FRAC = [0.4, 0.6, 0.0, 0.0, 0.0]  # [-] A, B, C, D, solvent
CRYST_MASS_FRAC = [0.0, 0.0, 0.2, 0.0, 0.8]  # [-]
SOLID_MASS_FRAC = [0.0, 0.0, 1.0, 0.0, 0.0]  # [-], pure C
TEMP = 313.0  # [K]
SIZE_GRID = np.arange(1.0, 200.0)  # [um]
VESSEL_DIAMETER = 0.01  # [m]
# Case-study crystallization kinetics, as in the batch-crystallizer module.
APELBLAT_COEFFS = np.array([-28.13909202, 0.001, 5.900800253])  # [-, K, -]
NUCL_PRIM = (6.26855218e18, 0.0, 8.30671806)
NUCL_SEC = (0.0, 0.0, 0.0, 0.0)
GROWTH = (1.45782420e6, 0.0, 4.52241037, 1.0, 3.93676056)


def _crystallizer(integrator):
    """Return an unseeded batch crystallizer, its structure compiled.

    Parameters
    ----------
    integrator : IntegratorBackend
        Backend handed to the vessel.

    Returns
    -------
    BatchCrystallizer
        Vessel at ``TEMP`` with a 1-D FVM population on ``SIZE_GRID``.
    """
    vessel = BatchCrystallizer(integrator=integrator, h_conv=0,
                               diam=VESSEL_DIAMETER, isothermal=True)
    solid = SolidPhase(DATA_PATH, mass=0, mass_frac=SOLID_MASS_FRAC, temp=TEMP)
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components="C", solvent_name="solvent",
        x_grid=SIZE_GRID, distrib_init=np.zeros_like(SIZE_GRID),
    )
    vessel.Phases = [
        LiquidPhase(DATA_PATH, mass=CHARGE_MASS, mass_frac=CRYST_MASS_FRAC,
                    temp=TEMP),
        solid,
    ]
    vessel.CrystKinetics = CrystKinetics(
        APELBLAT_COEFFS, nucl_prim=NUCL_PRIM, nucl_sec=NUCL_SEC,
        growth=GROWTH, solubility_type="apelblat",
    )
    vessel.compile_structure()
    return vessel


def _reactor(integrator=None):
    """Return a batch reactor charged with A and B, its structure compiled.

    Parameters
    ----------
    integrator : IntegratorBackend, optional
        Backend handed to the vessel; None leaves the vessel's default.

    Returns
    -------
    BatchReactor
        Isothermal vessel running A + B --> C.
    """
    vessel = BatchReactor(integrator=integrator, isothermal=True)
    vessel.Phases = LiquidPhase(DATA_PATH, mass=CHARGE_MASS,
                                mass_frac=REACTOR_MASS_FRAC, temp=TEMP)
    vessel.RxnKinetics = RxnKinetics(
        path=DATA_PATH, rxn_list=["A + B --> C"],
        k_params=np.array([1e-2]), ea_params=np.array([1e2]),
    )
    vessel.compile_structure()
    return vessel


def test_caller_options_keep_the_default_step_cap():
    """Options that set only rtol add to the defaults instead of replacing them."""
    backend = AssimuloBackend(options={"rtol": CALLER_RTOL})

    assert backend.solver_options() == {"maxh": BACKEND_DEFAULT_CAP,
                                        "rtol": CALLER_RTOL}


def test_backends_do_not_share_their_options():
    """Changing one backend's options leaves a new backend's untouched."""
    AssimuloBackend().options["rtol"] = CALLER_RTOL

    assert "rtol" not in AssimuloBackend().options


def test_crystallizer_asks_for_its_step_cap():
    """configure_solver hands the crystallizer's cap to the backend."""
    backend = AssimuloBackend()
    vessel = _crystallizer(backend)

    vessel.configure_solver()

    cap = BatchCrystallizer.NUCLEATION_STEP_CAP  # [s]
    assert cap != BACKEND_DEFAULT_CAP
    assert backend.solver_options()["maxh"] == cap


def test_caller_step_cap_outranks_the_crystallizer():
    """An explicit maxh is the caller's choice and stays."""
    backend = AssimuloBackend(options={"maxh": CALLER_CAP})
    vessel = _crystallizer(backend)

    vessel.configure_solver()

    assert backend.solver_options()["maxh"] == CALLER_CAP


def test_scipy_backend_keeps_its_own_step_size():
    """SciPy records the crystallizer's request but sets no max_step.

    LSODA gave the same case-study answers with and without the cap, and BDF
    stalled more often with it.
    """
    backend = ScipyBackend(method="LSODA")
    vessel = _crystallizer(backend)

    vessel.configure_solver()

    assert backend.max_step_request == BatchCrystallizer.NUCLEATION_STEP_CAP
    assert "max_step" not in backend.translate_options()


@pytest.mark.assimulo
def test_vessel_without_integrator_gets_cvode():
    """With Assimulo installed, a vessel built without a backend gets CVode."""
    pytest.importorskip("assimulo")

    assert isinstance(_reactor().integrator, AssimuloBackend)


@pytest.mark.assimulo
def test_compiled_cvode_takes_the_merged_step_cap():
    """The CVode solver itself ends up with the cap the precedence gives.

    The crystallizer's cap arrives after the solver is built, from
    configure_solver; a reactor asks for none and keeps the default.
    """
    pytest.importorskip("assimulo")

    cases = (
        (_crystallizer, None, BatchCrystallizer.NUCLEATION_STEP_CAP),
        (_crystallizer, {"maxh": CALLER_CAP}, CALLER_CAP),
        (_reactor, {"rtol": CALLER_RTOL}, BACKEND_DEFAULT_CAP),
    )
    for build, options, expected in cases:
        backend = AssimuloBackend(options=options)
        backend.compile_integrator(build(backend), verbose=False)

        assert backend._solver.maxh == pytest.approx(expected)


# Long enough for A + B --> C to make visible progress at the fixture's rate.
FALLBACK_RUNTIME = 60.0  # [s]

# Child-process prelude: any import of Assimulo fails as if it were absent.
BLOCK_ASSIMULO = textwrap.dedent("""
    import sys

    class _BlockAssimulo:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "assimulo" or fullname.startswith("assimulo."):
                raise ModuleNotFoundError(
                    "No module named %r" % fullname, name=fullname)
            return None

    sys.meta_path.insert(0, _BlockAssimulo())
    """)


def test_vessel_without_integrator_falls_back_to_lsoda(tmp_path):
    """Without Assimulo, a vessel built without a backend gets LSODA and solves."""
    script = BLOCK_ASSIMULO + textwrap.dedent(f"""
        import numpy as np
        from PharmaPy.Kinetics import RxnKinetics
        from PharmaPy.Phases_Refactored import LiquidPhase
        from PharmaPy.Reactors_Refactored import BatchReactor

        vessel = BatchReactor(isothermal=True)
        vessel.Phases = LiquidPhase({DATA_PATH!r}, mass={CHARGE_MASS!r},
                                    mass_frac={REACTOR_MASS_FRAC!r}, temp={TEMP!r})
        vessel.RxnKinetics = RxnKinetics(
            path={DATA_PATH!r}, rxn_list=["A + B --> C"],
            k_params=np.array([1e-2]), ea_params=np.array([1e2]))
        vessel.solve_unit(runtime={FALLBACK_RUNTIME!r}, verbose=False)

        conc = np.asarray(vessel.result.mole_conc_liquid0)
        print(type(vessel.integrator).__name__, vessel.integrator.method,
              bool(np.all(np.isfinite(conc))), bool(conc[-1, 2] > 0))
        """)
    environment = dict(os.environ, MPLCONFIGDIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                            env=environment, capture_output=True, text=True,
                            check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-4:] == ["ScipyBackend", "LSODA", "True", "True"]
