"""Lazy internal constructors for the optional Assimulo solver backend.

The exported names are factory functions, not Assimulo class aliases, and
therefore are not suitable for ``isinstance`` checks or subclassing.
Module-level lazy attributes were rejected because ``from ... import`` would
resolve them eagerly at the existing model call sites; factories preserve those
imports without loading Assimulo.

Notes
-----
Previously generated MetaModeler templates may import ``CVode``, ``IDA``,
``Explicit_Problem``, and ``Implicit_Problem`` from this module. Keep these
private compatibility exports while new templates use :mod:`PharmaPy.solvers`.
"""

from importlib import import_module
from functools import lru_cache
from typing import Any, NoReturn

# Source: the supported backend pin documented in DEPENDENCIES.md and
# environment.yml. Keep all user-facing solver diagnostics synchronized here.
_SUPPORTED_ASSIMULO_VERSION = "3.4.3"
_INSTALL_MESSAGE = (
    "Assimulo is required for solver-backed PharmaPy simulations. "
    f"Install Assimulo {_SUPPORTED_ASSIMULO_VERSION} from conda-forge using "
    "environment.yml or "
    "provide a compatible local installation."
)
_BROKEN_INSTALL_MESSAGE = (
    "Assimulo is installed but could not be imported. Verify that Assimulo "
    f"{_SUPPORTED_ASSIMULO_VERSION} and its compiled SUNDIALS libraries are "
    "compatible with this Python environment."
)


def _raise_assimulo_import_error(
    module_name: str, exc: ImportError
) -> NoReturn:
    """Raise the user-facing error for an Assimulo import failure.

    Parameters
    ----------
    module_name : str
        Fully qualified Assimulo module that failed to import.
    exc : ImportError
        Import failure raised by Python's module loader.

    Raises
    ------
    ImportError
        Actionable missing-install or broken-install error, chained to ``exc``.
    """
    if isinstance(exc, ModuleNotFoundError) and exc.name == "assimulo":
        raise ImportError(_INSTALL_MESSAGE) from exc

    raise ImportError(
        f"{_BROKEN_INSTALL_MESSAGE} Failed while importing {module_name}."
    ) from exc


def _load_assimulo_symbol(module_name: str, symbol_name: str) -> Any:
    """Load a symbol from the optional Assimulo dependency.

    Parameters
    ----------
    module_name : str
        Fully qualified Assimulo module name.
    symbol_name : str
        Attribute to load from the module.

    Returns
    -------
    Any
        Requested Assimulo class or callable.

    Raises
    ------
    ImportError
        If Assimulo or one of its required runtime components cannot be
        imported.
    """
    try:
        module = import_module(module_name)
    except ImportError as exc:
        _raise_assimulo_import_error(module_name, exc)

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(
            f"{_INSTALL_MESSAGE} The installed version does not provide "
            f"{module_name}.{symbol_name}."
        ) from exc


def _construct_assimulo_object(
    module_name: str,
    symbol_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Construct an object from the optional Assimulo dependency.

    Parameters
    ----------
    module_name : str
        Fully qualified Assimulo module name.
    symbol_name : str
        Constructor name within ``module_name``.
    *args : Any
        Positional arguments forwarded unchanged to the constructor.
    **kwargs : Any
        Keyword arguments forwarded unchanged to the constructor.

    Returns
    -------
    Any
        Constructed Assimulo problem or solver object.

    Raises
    ------
    ImportError
        If the requested Assimulo constructor is unavailable.
    """
    constructor = _load_assimulo_symbol(module_name, symbol_name)
    return constructor(*args, **kwargs)


class _CallbackFailure:
    """Latch the first model error during an outermost native solver operation.

    The native solver may retry at earlier times after a Python exception.
    Once a callback fails, all retries raise without re-entering the model.
    Calls outside a solver operation keep their ordinary Python behavior.
    """

    def __init__(self, callback: Any, solver_exceptions: tuple) -> None:
        """Store the callback and backend control-flow exception classes.

        Parameters
        ----------
        callback : callable
            RHS or residual; arguments, units and shapes are model-specific.
        solver_exceptions : tuple of type
            Explicitly recoverable callback exceptions to pass through.
            Backend errors from nested solves and termination requests raised
            by a model are failures of this callback, not outer solver control
            flow. Event handlers are outside this wrapper.
        """
        self.callback = callback
        self.solver_exceptions = solver_exceptions
        self.active = False
        self.error = None
        self.traceback = None
        self.diagnostic = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the model unless an earlier callback already failed.

        Parameters
        ----------
        *args, **kwargs : Any
            Original callback arguments, with their original units/shapes.

        Returns
        -------
        Any
            Original RHS or residual result, unchanged in units and shape.

        Raises
        ------
        BaseException
            First model error, with its saved traceback on every retry.
            Only explicit ``AssimuloRecoverableError`` requests pass through
            without latching. A one-shot ordinary numerical error is terminal.
        """
        if self.active and self.error is not None:
            raise self.error.with_traceback(self.traceback)
        try:
            return self.callback(*args, **kwargs)
        except self.solver_exceptions:
            raise
        except BaseException as error:
            if self.active:
                self.error = error
                self.traceback = error.__traceback__
            raise


class _FailureHandlingSolver:
    """Native solver mixin that exposes latched callback errors to Python."""

    def __init__(self, problem: Any, *args: Any, **kwargs: Any) -> None:
        """Give the native solver a private guarded callback.

        Parameters
        ----------
        problem : assimulo.problem.Problem
            Explicit or implicit problem, with model-specific units/shapes.
        *args, **kwargs : Any
            Arguments forwarded to the native solver constructor.

        Notes
        -----
        Assimulo 3.4.3 caches the callback at construction. Restore the
        problem's original callable immediately afterwards, so sharing a
        problem between solvers does not share failure state or alter calls
        made directly to the problem.
        """
        callback_name = self._callback_name
        callback = getattr(problem, callback_name)
        self._callback_failure = _CallbackFailure(callback, (
            # Only this callback exception requests native recovery in the
            # supported Assimulo backend. Native errors from a nested solve
            # and event exceptions raised by a model are callback failures.
            _load_assimulo_symbol("assimulo.exception", "AssimuloRecoverableError"),
        ))
        setattr(problem, callback_name, self._callback_failure)
        try:
            super().__init__(problem, *args, **kwargs)
        finally:
            setattr(problem, callback_name, callback)

    def _run(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a native operation with a fresh, exception-safe failure latch.

        Parameters
        ----------
        operation : callable
            Native ``simulate`` or IDA ``make_consistent`` method.
        *args, **kwargs : Any
            Native operation arguments, retaining backend units/shapes.

        Returns
        -------
        Any
            Native result, unchanged.

        Raises
        ------
        BaseException
            Original callback error, chained from the backend diagnostic.
            Without a recorded model failure, native errors pass unchanged.

        Notes
        -----
        Only the outermost operation clears the shared scope. In particular,
        IDA event handlers may call ``make_consistent`` before ``simulate``
        resumes. Retain an inner operation's native diagnostic if the original
        exception bubbles through the outer operation or the handler catches
        it. Clear all records even for interrupts when the outer operation
        ends, so later solves neither retain a failed model nor inherit its
        error.
        """
        failure = self._callback_failure
        owns_scope = not failure.active
        failure.active = True
        try:
            try:
                result = operation(*args, **kwargs)
            except BaseException as backend_error:
                if failure.error is not None:
                    if backend_error is not failure.error:
                        failure.diagnostic = backend_error
                    if failure.diagnostic is not None:
                        raise failure.error.with_traceback(
                            failure.traceback) from failure.diagnostic
                raise
            if failure.error is not None:
                if operation.__name__ == "make_consistent":
                    # Assimulo 3.4.3 returns [IDA flag, y, yd] on a
                    # consistency failure instead of raising IDAError.
                    native_error = _load_assimulo_symbol(
                        "assimulo.solvers.sundials", "IDAError")
                    failure.diagnostic = native_error(result[0], self.t)
                elif failure.diagnostic is None:
                    # Defensive: event handling or a future backend must
                    # not return success after a model callback failed.
                    failure.diagnostic = RuntimeError(
                        "Assimulo returned after a model callback failed")
                raise failure.error.with_traceback(
                    failure.traceback) from failure.diagnostic
            return result
        finally:
            if owns_scope:
                failure.active = False
                failure.error = None
                failure.traceback = None
                failure.diagnostic = None

    def simulate(self, *args: Any, **kwargs: Any) -> Any:
        """Integrate while preserving the first RHS or residual exception.

        Parameters
        ----------
        *args, **kwargs : Any
            Native ``simulate`` arguments: final time [s] and optional
            reporting grid [s] (or the model's independent-coordinate units).

        Returns
        -------
        tuple
            Native times, states and, for IDA, derivatives, with the model's
            original units and shapes. Normal event termination is preserved.

        Raises
        ------
        BaseException
            Original callback failure (including KeyboardInterrupt and
            SystemExit), chained from the native error; otherwise the native
            error unchanged.
        """
        return self._run(super().simulate, *args, **kwargs)


class _ImplicitFailureHandlingSolver(_FailureHandlingSolver):
    """Also guard residuals evaluated during IDA consistent initialization."""

    def make_consistent(self, *args: Any, **kwargs: Any) -> Any:
        """Compute consistent initial states with residual-error propagation.

        Parameters
        ----------
        *args, **kwargs : Any
            Native IDA consistency arguments, including the mode string.

        Returns
        -------
        Any
            Native IDA consistency result, unchanged.

        Raises
        ------
        BaseException
            Original residual failure chained from the backend error, or a
            native error unchanged when no model callback failed.
        """
        return self._run(super().make_consistent, *args, **kwargs)


@lru_cache(maxsize=None)
def _solver_class(name: str) -> type:
    """Build a guarded native solver subclass without eager optional imports.

    Parameters
    ----------
    name : {'CVode', 'IDA'}
        Backend solver class to load.

    Returns
    -------
    type
        Native solver subclass sharing the callback-failure contract.

    Raises
    ------
    ImportError
        If the optional backend constructor cannot be loaded.
    """
    native = _load_assimulo_symbol("assimulo.solvers", name)
    mixin = _FailureHandlingSolver if name == "CVode" else _ImplicitFailureHandlingSolver
    callback_name = "rhs" if name == "CVode" else "res"
    return type(name, (mixin, native), {"_callback_name": callback_name})


def CVode(*args: Any, **kwargs: Any) -> Any:
    """Construct an Assimulo CVode solver on first use.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to ``assimulo.solvers.CVode``.
    **kwargs : Any
        Keyword arguments forwarded to ``assimulo.solvers.CVode``.

    Returns
    -------
    Any
        Native CVode subclass with per-solve callback-failure handling.

    Raises
    ------
    ImportError
        If the optional Assimulo backend is unavailable.
    """
    return _solver_class("CVode")(*args, **kwargs)


def IDA(*args: Any, **kwargs: Any) -> Any:
    """Construct an Assimulo IDA solver on first use.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to ``assimulo.solvers.IDA``.
    **kwargs : Any
        Keyword arguments forwarded to ``assimulo.solvers.IDA``.

    Returns
    -------
    Any
        Native IDA subclass guarding residuals during ``simulate`` and
        ``make_consistent``.

    Raises
    ------
    ImportError
        If the optional Assimulo backend is unavailable.
    """
    return _solver_class("IDA")(*args, **kwargs)


def Explicit_Problem(*args: Any, **kwargs: Any) -> Any:
    """Construct an Assimulo explicit problem on first use.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to
        ``assimulo.problem.Explicit_Problem``.
    **kwargs : Any
        Keyword arguments forwarded to
        ``assimulo.problem.Explicit_Problem``.

    Returns
    -------
    Any
        Assimulo explicit-problem instance.

    Raises
    ------
    ImportError
        If the optional Assimulo backend is unavailable.
    """
    return _construct_assimulo_object(
        "assimulo.problem", "Explicit_Problem", *args, **kwargs
    )


def Implicit_Problem(*args: Any, **kwargs: Any) -> Any:
    """Construct an Assimulo implicit problem on first use.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to
        ``assimulo.problem.Implicit_Problem``.
    **kwargs : Any
        Keyword arguments forwarded to
        ``assimulo.problem.Implicit_Problem``.

    Returns
    -------
    Any
        Assimulo implicit-problem instance.

    Raises
    ------
    ImportError
        If the optional Assimulo backend is unavailable.
    """
    return _construct_assimulo_object(
        "assimulo.problem", "Implicit_Problem", *args, **kwargs
    )
