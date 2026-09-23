"""Import-boundary regressions for the optional Assimulo solver stack.

Absent-backend cases block Assimulo imports inside isolated child processes so
they run in every development environment. The core CI lane also asserts that
Assimulo is genuinely absent, while Assimulo-marked cases exercise the
installed backend's malformed-module and missing-symbol paths.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import PharmaPy._assimulo as assimulo_backend

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
AFFECTED_MODULES = (
    "PharmaPy.Containers",
    "PharmaPy.Crystallizers",
    "PharmaPy.Distillation",
    "PharmaPy.Drying_Model",
    "PharmaPy.DynamicExtraction",
    "PharmaPy.Evaporators",
    "PharmaPy.Reactors",
    "PharmaPy.SolidLiquidSep",
    "PharmaPy.ThreePhaseSettler",
)
LAZY_CONSTRUCTORS = (
    "CVode",
    "IDA",
    "Explicit_Problem",
    "Implicit_Problem",
)

_ASSIMULO_IMPORT_BLOCKER = textwrap.dedent('''
    import sys

    class _BlockAssimulo:
        """Reject Assimulo imports in this child process only."""

        @staticmethod
        def find_spec(fullname, path=None, target=None):
            """Reject module specifications for the Assimulo package.

            Parameters
            ----------
            fullname : str
                Fully qualified name of the requested module.
            path : sequence of str or None, optional
                Parent package search path supplied by the import system.
            target : module or None, optional
                Existing module supplied when resolving a reload.

            Returns
            -------
            None
                Signal that unrelated modules should use later finders.

            Raises
            ------
            ModuleNotFoundError
                If Assimulo or one of its submodules is requested.
            """
            if fullname == "assimulo" or fullname.startswith("assimulo."):
                raise ModuleNotFoundError(
                    f"No module named {fullname!r}", name=fullname
                )
            return None

    sys.meta_path.insert(0, _BlockAssimulo())
    ''')


def _run_with_assimulo_blocked(script, tmp_path):
    """Run Python source with Assimulo blocked in the child process.

    Parameters
    ----------
    script : str
        Python source to execute in the child process.
    tmp_path : pathlib.Path
        Temporary directory used for the Matplotlib configuration cache.

    Returns
    -------
    subprocess.CompletedProcess
        Completed child-process result with captured text output.

    Notes
    -----
    The process-local import blocker leaves the parent interpreter unchanged,
    so absent-backend regressions run whether or not Assimulo is installed.
    The locked core CI lane separately verifies behavior under genuine absence.
    """
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path)
    blocked_script = f"{_ASSIMULO_IMPORT_BLOCKER}\n{script}"
    return subprocess.run(
        [sys.executable, "-c", blocked_script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_model_modules_import_without_assimulo(tmp_path):
    """Every solver-backed model module imports without the optional backend."""
    script = textwrap.dedent(f"""
        import importlib

        for module_name in {AFFECTED_MODULES!r}:
            importlib.import_module(module_name)
        """)
    result = _run_with_assimulo_blocked(script, tmp_path)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("symbol_name", LAZY_CONSTRUCTORS)
def test_solver_construction_reports_missing_assimulo(symbol_name, tmp_path):
    """Lazy solver construction gives an actionable dependency error."""
    script = textwrap.dedent(f"""
        from PharmaPy._assimulo import {symbol_name}

        try:
            {symbol_name}(object())
        except ImportError as exc:
            expected = (
                "Assimulo is required for solver-backed PharmaPy simulations"
            )
            if expected not in str(exc):
                raise AssertionError(str(exc)) from exc
            # Both the missing-install and the incompatible-install branch embed
            # the message above, so assert the discriminator too: a missing
            # install is reported from ModuleNotFoundError, while an installed
            # backend lacking the symbol is reported from AttributeError and
            # adds "does not provide". Without this the test would still pass
            # while exercising the branch it is not named for.
            if not isinstance(exc.__cause__, ModuleNotFoundError):
                raise AssertionError(
                    "expected a missing-install error, got cause "
                    f"{{type(exc.__cause__).__name__}}: {{exc}}"
                ) from exc
        else:
            raise AssertionError(
                "{symbol_name} construction unexpectedly succeeded"
            )
        """)
    result = _run_with_assimulo_blocked(script, tmp_path)

    assert result.returncode == 0, result.stderr


@pytest.mark.assimulo
def test_missing_assimulo_symbol_reports_qualified_name():
    """The real backend identifies a requested constructor it lacks."""
    pytest.importorskip("assimulo.solvers")
    missing_symbol = "ConstructorThatAssimuloMustNotProvide"

    with pytest.raises(
        ImportError,
        match=rf"assimulo\.solvers\.{missing_symbol}",
    ) as exc_info:
        assimulo_backend._construct_assimulo_object(
            "assimulo.solvers", missing_symbol
        )

    assert isinstance(exc_info.value.__cause__, AttributeError)


@pytest.mark.assimulo
def test_broken_assimulo_install_has_distinct_error():
    """Loader failures classify as broken rather than missing installs."""
    pytest.importorskip("assimulo.solvers")
    missing_module = "assimulo.solvers.no_such_submodule"

    with pytest.raises(
        ImportError,
        match="Assimulo is installed but could not be imported",
    ) as exc_info:
        assimulo_backend._load_assimulo_symbol(
            missing_module, "CVode"
        )

    assert missing_module in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
