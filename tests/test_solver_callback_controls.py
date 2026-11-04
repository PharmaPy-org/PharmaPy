"""Native CVode/IDA controls and solver reuse for the #435 shared guard.

Analytic first-order decay exercises real Assimulo callbacks, consistent IDA
initialization and root termination. No native or PharmaPy object is replaced.
The public-unit failure and subprocess safety probes live in the companion
``test_solver_callback_failures`` module.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip("assimulo")

from assimulo.exception import AssimuloRecoverableError, TerminateSimulation
from assimulo.solvers.sundials import CVodeError, IDAError
from PharmaPy.solvers import CVode, Explicit_Problem, IDA, Implicit_Problem

# Synthetic single-species decay over one rate-constant time scale.
RATE = 1.0  # [1/s]
CHARGE = np.array([1.0])  # [mol/L]
END = 1.0 / RATE  # [s]
STOP = END / 2  # [s], an interior event root
# Ten times the native default rtol accommodates accumulated local errors;
# this is a control of unmodified successful integration, not new accuracy policy.
TRAJECTORY_RTOL = 1e-5  # [-]
# Sixteen epsilons at the run's time scale allow rounding of the few
# arithmetic operations in the linear root finder (test design assumption).
EVENT_ATOL = 16 * np.finfo(float).eps * END  # [s]
MAX_CALLBACKS = 2000  # [-], same safety budget as the public-unit probes


class Decay:
    """First-order decay with an optional genuine user callback failure."""

    def __init__(self, error=None):
        """Store an optional failure and the callback evaluation count.

        Parameters
        ----------
        error : BaseException or None, optional
            Error raised by the model when enabled, at any evaluation time.
        """
        self.error = error
        self.calls = 0

    def rhs(self, time, states):
        """Evaluate the analytic single-species decay RHS.

        Parameters
        ----------
        time : float
            Time [s].
        states : ndarray
            Concentration [mol/L], shape (1,).

        Returns
        -------
        ndarray
            Concentration derivative [mol/L/s], shape (1,).

        Raises
        ------
        BaseException
            Configured model error, when enabled.
        """
        self.calls += 1
        if self.error is not None:
            raise self.error
        return -RATE * states

    def residual(self, time, states, derivatives):
        """Evaluate the equivalent differential residual for IDA.

        Parameters
        ----------
        time : float
            Time [s].
        states : ndarray
            Concentration [mol/L], shape (1,).
        derivatives : ndarray
            Concentration derivative [mol/L/s], shape (1,).

        Returns
        -------
        ndarray
            Derivative minus decay rate [mol/L/s], shape (1,).

        Raises
        ------
        BaseException
            Configured model error, when enabled.
        """
        return derivatives - self.rhs(time, states)


def _problem(family, decay):
    """Construct a native problem in concentration units.

    Parameters
    ----------
    family : {'CVode', 'IDA'}
        Native solver family.
    decay : Decay
        Analytic user model.

    Returns
    -------
    tuple
        Real problem, solver factory, callback attribute and native error type.
    """
    if family == "CVode":
        return Explicit_Problem(decay.rhs, CHARGE), CVode, "rhs", CVodeError
    return (Implicit_Problem(decay.residual, CHARGE, -RATE * CHARGE),
            IDA, "res", IDAError)


@pytest.mark.parametrize("family", ["CVode", "IDA"])
def test_same_solver_clears_failure_and_keeps_problem_callback(family):
    decay = Decay(ValueError("original decay failure"))
    problem, factory, callback_name, error_type = _problem(family, decay)
    callback = getattr(problem, callback_name)
    solver = factory(problem)
    solver.verbosity = 50
    # Two solvers sharing a problem retain independent failure state, and
    # the problem's callable remains usable directly with its native meaning.
    other = factory(problem)
    other.verbosity = 50
    assert getattr(problem, callback_name) is callback
    with pytest.raises(ValueError, match="original decay failure") as failure:
        solver.simulate(END)
    assert failure.value is decay.error
    assert isinstance(failure.value.__cause__, error_type)
    assert decay.calls == 1
    assert solver.statistics["nfcns"] < MAX_CALLBACKS
    decay.error = None
    solver.reset()
    time, states, *_ = solver.simulate(END)  # [s], [mol/L], optional [mol/L/s]
    other_time, other_states, *_ = other.simulate(END)  # [s], [mol/L], optional [mol/L/s]
    np.testing.assert_allclose(states[-1], CHARGE * np.exp(-RATE * END),
                               rtol=TRAJECTORY_RTOL)
    np.testing.assert_array_equal(time, other_time)
    np.testing.assert_array_equal(states, other_states)


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_ida_consistent_initialization_preserves_residual_error(error_type):
    decay = Decay(error_type("inconsistent feed model"))
    problem, factory, _, native_error = _problem("IDA", decay)
    solver = factory(problem)
    solver.verbosity = 50
    with pytest.raises(error_type, match="inconsistent feed model") as failure:
        solver.make_consistent("IDA_YA_YDP_INIT")
    assert failure.value is decay.error
    assert isinstance(failure.value.__cause__, native_error)
    assert decay.calls == 1
    decay.error = None
    solver.make_consistent("IDA_YA_YDP_INIT")
    _, states, _ = solver.simulate(END)  # [mol/L], state profile
    np.testing.assert_allclose(states[-1], CHARGE * np.exp(-RATE * END),
                               rtol=TRAJECTORY_RTOL)


@pytest.mark.parametrize("family", ["CVode", "IDA"])
def test_event_termination_keeps_analytic_trajectory(family):
    problem, factory, _, _ = _problem(family, Decay())
    problem.state_events = lambda time, states, *args: np.array([time - STOP])

    def terminate(solver, event_info):
        """Stop normally at the one linear time root.

        Parameters
        ----------
        solver : object
            Running native solver with time [s] and concentration [mol/L].
        event_info : tuple
            State-root flags [-] and the time-event flag.

        Raises
        ------
        TerminateSimulation
            Normal Assimulo event termination.
        """
        assert event_info[0][0] != 0
        raise TerminateSimulation

    problem.handle_event = terminate
    solver = factory(problem)
    solver.verbosity = 50
    time, states, *_ = solver.simulate(END)  # [s], [mol/L], optional [mol/L/s]
    assert time[-1] == pytest.approx(STOP, abs=EVENT_ATOL, rel=0)
    np.testing.assert_allclose(states[-1], CHARGE * np.exp(-RATE * STOP),
                               rtol=TRAJECTORY_RTOL)


@pytest.mark.parametrize("family", ["CVode", "IDA"])
def test_native_failure_is_not_relabelled(family):
    problem, factory, _, error_type = _problem(family, Decay())
    solver = factory(problem)
    solver.verbosity = 50
    # One internal step is deliberately insufficient for a decay time scale.
    solver.maxsteps = 1  # [-]
    with pytest.raises(error_type) as failure:
        solver.simulate(END, ncp_list=[END])
    assert failure.value.__cause__ is None


def test_explicit_recoverable_solver_exception_is_not_latched():
    recovered = []  # [s], time of the one deliberate recoverable failure

    def rhs(time, states):
        """Request one recoverable retry during analytic decay.

        Parameters
        ----------
        time : float
            Time [s].
        states : ndarray
            Concentration [mol/L], shape (1,).

        Returns
        -------
        ndarray
            Decay derivative [mol/L/s], shape (1,).

        Raises
        ------
        AssimuloRecoverableError
            One intentional backend retry, after integration has started.
        """
        if time > STOP and not recovered:
            recovered.append(time)
            raise AssimuloRecoverableError("retry this trial")
        return -RATE * states

    solver = CVode(Explicit_Problem(rhs, CHARGE))
    solver.verbosity = 50
    _, states = solver.simulate(END)  # [mol/L], state profile
    assert len(recovered) == 1
    np.testing.assert_allclose(states[-1], CHARGE * np.exp(-RATE * END),
                               rtol=TRAJECTORY_RTOL)


@pytest.mark.parametrize("failure_phase", [
    None, "after_event", "during_consistency", "caught_consistency",
], ids=["healthy", "later-residual", "inner-failure", "caught-inner-failure"])
def test_ida_nested_consistency_preserves_outer_failure_scope(failure_phase):
    """Consistency initialization inside an event belongs to the same solve.

    The supported Assimulo 3.4.3 ``examples/ida_with_disc.py`` calls
    ``make_consistent`` from an event handler before integration resumes.
    Exercise that real lifecycle, including an inner failure that the
    handler catches before requesting normal termination.

    Parameters
    ----------
    failure_phase : str or None
        Fail after the event, inside consistency initialization, or inside
        initialization with the error caught by the handler; None is healthy.
    """
    decay = Decay()
    error = ValueError("nested event residual failure")
    enabled = {"failure": failure_phase is not None}
    events = []  # [s], event-handler evaluation times
    failure_time = (STOP + END) / 2  # [s], between event and requested end

    def residual(time, states, derivatives, switches=None):
        """Evaluate decay, optionally failing after successful event handling.

        Parameters
        ----------
        time : float
            Time [s].
        states, derivatives : ndarray
            Concentration [mol/L] and its derivative [mol/L/s], shape (1,).
        switches : list of bool or None, optional
            Root-enable switch; the decay equation does not depend on it.

        Returns
        -------
        ndarray
            Decay residual [mol/L/s], shape (1,).

        Raises
        ------
        ValueError
            Configured failure inside or after the event's initialization.
        """
        if enabled["failure"] and failure_phase == "after_event" and time > failure_time:
            raise error
        return decay.residual(time, states, derivatives)

    def initialize_at_event(solver, event_info):
        """Recompute consistent states once, as Assimulo's example does.

        Parameters
        ----------
        solver : object
            Running IDA solver with time [s] and concentration [mol/L].
        event_info : tuple
            State-root flags [-] and the time-event flag.

        Raises
        ------
        ValueError
            Original model failure from nested consistent initialization.
        TerminateSimulation
            Handler's termination request after catching a model failure.
        """
        events.append(solver.t)
        solver.sw[0] = False
        if enabled["failure"] and failure_phase in ("during_consistency", "caught_consistency"):
            decay.error = error
        try:
            solver.make_consistent("IDA_YA_YDP_INIT")
        except ValueError as caught:
            if failure_phase != "caught_consistency":
                raise
            assert caught is error
            raise TerminateSimulation

    problem = Implicit_Problem(residual, CHARGE, -RATE * CHARGE, sw0=[True])
    problem.state_events = lambda time, states, derivatives, switches: np.array(
        [time - STOP if switches[0] else END])
    problem.handle_event = initialize_at_event
    solver = IDA(problem)
    solver.verbosity = 50
    if enabled["failure"]:
        with pytest.raises(ValueError, match="nested event residual failure") as caught:
            solver.simulate(END)
        assert caught.value is error
        assert isinstance(error.__cause__, IDAError)
        assert error.__cause__ is not error
        assert len(events) == 1
        assert solver.statistics["nfcns"] < MAX_CALLBACKS
        enabled["failure"] = False
        decay.error = None
        events.clear()
        solver.reset()
    _, states, _ = solver.simulate(END)  # [mol/L], state profile
    assert len(events) == 1
    np.testing.assert_allclose(states[-1], CHARGE * np.exp(-RATE * END),
                               rtol=TRAJECTORY_RTOL)
