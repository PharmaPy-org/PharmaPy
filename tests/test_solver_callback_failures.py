"""Real CVode/IDA callback failures through public unit solves (#435).

Each failure probe runs in a child process: unchanged CVode can stall at a
delayed input failure, including KeyboardInterrupt. The child inherits the
selected interpreter, limits address space/CPU on POSIX, and exits explicitly
if the input budget is exhausted. No solver or unit collaborator is replaced.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# Standalone child probes must import this complete tree, including when
# run in a disposable mutation checkout with another editable install present.
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip("assimulo")

from assimulo.exception import TerminateSimulation
from assimulo.solvers.sundials import CVodeError, IDAError

from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Reactors import CSTR
from PharmaPy.DynamicExtraction import DynamicExtractor
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream
from test_reactor_correctness import _configured_reactor, CONCENTRATIONS
from test_solver_callback_controls import Decay, END, _problem
from test_dynamic_extractor_inlet_temperatures import (
    SPECIES, LIGHT_COMPOSITION, HEAVY_COMPOSITION, LIGHT_FLOW, HEAVY_FLOW,
    PHASE_HOLDUP, INITIAL_TEMPERATURE, distribution_coefficients,
)

# Synthetic one-second run with onset at the first positive evaluation or
# one quarter of the run. Neither threshold changes physical model parameters.
RUNTIME = 1.0  # [s]
DELAY = RUNTIME / 4  # [s]
# Far above the few hundred input evaluations of these small healthy units;
# below the issue's 5000-call stall probe. Counts are a safety/contract bound,
# not a tuning parameter of the integrator.
MAX_INPUT_CALLS = 2000  # [-]
# Task-specific child exit status, distinct from Python's ordinary failure.
INPUT_BUDGET_EXIT = 23  # [-]
PROBE_TIMEOUT = 60  # [s], includes interpreter/import startup
PROBE_CPU_SECONDS = 30  # [s], secondary guard against a native retry loop
PROBE_MEMORY_BYTES = 6 * 1024**3  # [byte], ample for small NumPy/Assimulo cases
EXCEPTIONS = (ValueError, AttributeError, KeyError, KeyboardInterrupt, SystemExit)
BACKEND_MODEL_EXCEPTIONS = (TerminateSimulation, CVodeError, IDAError)
ONE_SHOT_EXCEPTIONS = (ValueError, np.linalg.LinAlgError, ZeroDivisionError)
CALLBACK_EXCEPTIONS = EXCEPTIONS + BACKEND_MODEL_EXCEPTIONS + ONE_SHOT_EXCEPTIONS


def _callback_error(error_type):
    """Construct a model error, obtaining native diagnostics from a real solve.

    Parameters
    ----------
    error_type : type
        Model exception class; CVodeError and IDAError represent a nested
        solver failure that the outer model subsequently raises.

    Returns
    -------
    BaseException
        Original error object to raise at the delayed feed handoff. Native
        diagnostics use the backend's actual code, message and time [s].

    Raises
    ------
    AssertionError
        If one allowed native step unexpectedly completes one decay time scale.

    Notes
    -----
    Generate a genuine too-much-work diagnostic rather than inventing a
    SUNDIALS status code. The native-control test uses this same analytic
    problem and explicit maxsteps=1 constraint.
    """
    if error_type in (CVodeError, IDAError):
        family = "CVode" if error_type is CVodeError else "IDA"
        problem, factory, _, _ = _problem(family, Decay())
        nested_solver = factory(problem)
        nested_solver.verbosity = 50
        # One step cannot resolve a full first-order decay time scale.
        nested_solver.maxsteps = 1  # [-], deliberate native work-budget failure
        try:
            nested_solver.simulate(END, ncp_list=[END])
        except error_type as error:
            assert error.__cause__ is None
            return error
        raise AssertionError("nested solver unexpectedly met its work budget")
    return error_type("feed calibration unavailable: species A")


class FailingInput:
    """A real DynamicInput feed that fails at a prescribed time.

    The one-shot mode represents a callback whose next evaluation could
    succeed. A solve must still stop on the first failure.
    """

    def __init__(self, values, error_type, onset, one_shot=False):
        """Store feed composition and failure configuration.

        Parameters
        ----------
        values : array_like
            CSTR concentrations [mol/L] or extractor mole fractions [-].
        error_type : type
            Exception class to raise from the feed.
        onset : float
            Last time [s] at which the feed is valid.
        one_shot : bool, optional
            Allow all evaluations after the first failure to succeed.
        """
        self.values = np.array(values)  # [mol/L] or [-], as supplied
        self.error = _callback_error(error_type)
        self.onset = onset  # [s]
        self.one_shot = one_shot
        self.enabled = True
        self.calls = 0
        self.calls_after_failure = 0
        self.first_traceback = None

    def __call__(self, time):
        """Evaluate the feed or raise its original configured exception.

        Parameters
        ----------
        time : float or ndarray
            Evaluation time [s].

        Returns
        -------
        ndarray
            Composition [mol/L] or [-], with time axes followed by species.

        Raises
        ------
        BaseException
            The configured input error after ``onset`` while enabled.
        """
        self.calls += 1
        if self.calls > MAX_INPUT_CALLS:
            print("INPUT BUDGET EXHAUSTED", flush=True)
            os._exit(INPUT_BUDGET_EXIT)
        if self.enabled and self.first_traceback is not None:
            self.calls_after_failure += 1
        if (self.enabled and np.any(np.asarray(time) > self.onset)
                and not (self.one_shot and self.first_traceback is not None)):
            try:
                raise self.error
            except BaseException as error:
                if self.first_traceback is None:
                    self.first_traceback = error.__traceback__
                raise
        return np.broadcast_to(self.values, np.shape(time) + self.values.shape).copy()


def _unit(family, path):
    """Build real CSTR or two-stage extractor collaborators.

    Parameters
    ----------
    family : {'CVode', 'IDA'}
        Unit's native integrator.
    path : Path
        Temporary directory for the synthetic extractor database.

    Returns
    -------
    tuple
        Unit, controlled inlet, input field name and composition ([mol/L]
        for CSTR, mole fractions [-] for extractor).
    """
    if family == "CVode":
        unit = _configured_reactor(CSTR(isothermal=False))
        return unit, unit.Inlet, "mole_conc", CONCENTRATIONS
    database = path / "extraction.json"
    database.write_text(json.dumps(SPECIES))
    unit = DynamicExtractor(num_stages=2, k_fun=distribution_coefficients,
                            gamma_model="ideal")
    unit.Phases = LiquidPhase(
        str(database), mole_frac=(LIGHT_COMPOSITION + HEAVY_COMPOSITION) / 2,
        moles=2 * PHASE_HOLDUP, temp=INITIAL_TEMPERATURE, verbose=False)
    unit.Inlet = {
        "feed": LiquidStream(str(database), mole_frac=LIGHT_COMPOSITION,
                             mole_flow=LIGHT_FLOW, temp=INITIAL_TEMPERATURE,
                             verbose=False),
        "solvent": LiquidStream(str(database), mole_frac=HEAVY_COMPOSITION,
                                mole_flow=HEAVY_FLOW, temp=INITIAL_TEMPERATURE,
                                verbose=False),
    }
    return unit, unit.Inlet["feed"], "mole_frac", LIGHT_COMPOSITION


def _probe(family, error_name, onset, one_shot):
    """Check propagation, bounded calls and subsequent public-solve cleanup.

    Parameters
    ----------
    family : {'CVode', 'IDA'}
        Native unit integrator.
    error_name : str
        Name of the exception to raise.
    onset : float
        Valid-feed endpoint [s].
    one_shot : bool
        Whether the feed could recover after its first failure.

    Raises
    ------
    AssertionError
        If identity, message, traceback, solver cause, latching or cleanup
        differs from the shared failure contract.

    Notes
    -----
    CSTR states contain concentrations [mol/L] and temperatures [K];
    extractor states contain mole fractions [-], energies [J] and
    temperatures [K]. Both public solves return time [s] first.
    """
    import PharmaPy
    assert Path(PharmaPy.__file__).resolve().parent.parent == ROOT
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        unit, inlet, field, values = _unit(family, path)
        error_type = next(cls for cls in CALLBACK_EXCEPTIONS
                          if cls.__name__ == error_name)
        feed = FailingInput(values, error_type, onset, one_shot)
        expected_message = str(feed.error)
        controller = DynamicInput()
        controller.add_variable(field, feed)
        inlet.DynamicInlet = controller
        try:
            unit.solve_unit(runtime=RUNTIME, verbose=False)
        except BaseException as error:
            assert error is feed.error, (type(error).__name__, str(error))
            assert str(error) == expected_message
            assert feed.calls_after_failure == 0, feed.calls_after_failure
            assert isinstance(error.__cause__, CVodeError if family == "CVode" else IDAError)
            assert error.__cause__ is not error
            traceback = error.__traceback__
            while traceback is not None and traceback is not feed.first_traceback:
                traceback = traceback.tb_next
            assert traceback is feed.first_traceback is not None
        else:
            raise AssertionError("solve stepped around the callback failure")
        assert feed.calls < MAX_INPUT_CALLS
        if family == "CVode":
            assert unit.elapsed_time == 0
        assert unit.profiles_runs == []
        assert not hasattr(unit, "result")
        feed.enabled = False
        if family == "CVode":
            # Reactor RHS evaluations already update phase temperature;
            # reset the charge through the existing public reset API.
            unit.reset()
        _, states = unit.solve_unit(runtime=RUNTIME, verbose=False)  # [state units]
        fresh, _, _, _ = _unit(family, path)
        _, expected = fresh.solve_unit(runtime=RUNTIME, verbose=False)  # [state units]
        # Identical models/initial states must follow the same trajectory;
        # exact equality guards leaked failure/solver state, not accuracy.
        np.testing.assert_array_equal(states, expected)
        print(f"PROPAGATED {family} {error_name}; input calls={feed.calls}", flush=True)


@pytest.mark.parametrize("family", ["CVode", "IDA"])
@pytest.mark.parametrize("onset", [0.0, DELAY], ids=["persistent", "delayed"])
@pytest.mark.parametrize("error_type", EXCEPTIONS, ids=lambda cls: cls.__name__)
def test_public_solve_preserves_callback_failure(family, onset, error_type):
    _run_probe(family, error_type.__name__, onset, False)


@pytest.mark.parametrize("family", ["CVode", "IDA"])
@pytest.mark.parametrize("error_type", BACKEND_MODEL_EXCEPTIONS,
                         ids=lambda cls: cls.__name__)
def test_public_solve_latches_backend_family_callback_failures(family, error_type):
    """Treat backend-family errors raised by the model as model failures.

    Parameters
    ----------
    family : {'CVode', 'IDA'}
        Outer public unit's native integrator.
    error_type : type
        Termination signal or genuine nested-solver error raised after DELAY [s].
    """
    _run_probe(family, error_type.__name__, DELAY, False)


@pytest.mark.parametrize("family", ["CVode", "IDA"])
@pytest.mark.parametrize("error_type", ONE_SHOT_EXCEPTIONS,
                         ids=lambda cls: cls.__name__)
def test_public_solve_cannot_step_around_one_shot_failure(family, error_type):
    """Abort on the original error even if a later callback could succeed.

    Parameters
    ----------
    family : {'CVode', 'IDA'}
        Public unit's native integrator.
    error_type : type
        ValueError, LinAlgError or ZeroDivisionError raised once after DELAY [s].

    Notes
    -----
    One-shot numerical errors are intentionally terminal for both solver
    families. A model that permits backend retry must request it explicitly
    with AssimuloRecoverableError instead of an ordinary numerical error.
    """
    _run_probe(family, error_type.__name__, DELAY, True)


def _run_probe(family, error_name, onset, one_shot):
    """Run the potentially hanging public solve in an isolated process.

    Parameters
    ----------
    family, error_name : str
        Integrator and exception names for the child probe.
    onset : float
        Failure threshold [s].
    one_shot : bool
        Whether the failing input could recover on retry.

    Raises
    ------
    AssertionError
        If the child fails or omits the successful contract sentinel.
    subprocess.TimeoutExpired
        If the public solve exceeds the wall-clock safety bound.
    """
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), family, error_name,
         str(onset), str(int(one_shot))], cwd=ROOT, capture_output=True,
        text=True, timeout=PROBE_TIMEOUT, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PROPAGATED {family} {error_name}" in result.stdout


if __name__ == "__main__":
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_AS,
                           (PROBE_MEMORY_BYTES, PROBE_MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU,
                           (PROBE_CPU_SECONDS, PROBE_CPU_SECONDS))
    _probe(sys.argv[1], sys.argv[2], float(sys.argv[3]), bool(int(sys.argv[4])))
