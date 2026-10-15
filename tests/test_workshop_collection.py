"""Check notebook-test collection with individual optional tools unavailable.

These solver-lane cases use real installed dependencies and block one import
inside a child process. They exercise collection only; the separate notebook
regression covers execution with all tools present.
"""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_TOOLS = ("nbformat", "nbclient", "jupyter_client")


@pytest.mark.parametrize("missing_module", NOTEBOOK_TOOLS)
def test_workshop_collection_without_notebook_tool(missing_module: str) -> None:
    """Keep a core test runnable when one notebook import is unavailable.

    Parameters
    ----------
    missing_module : str
        Optional notebook package to make unavailable in the child process.

    Notes
    -----
    Preload the real packages before blocking the selected module so transitive
    imports cannot mask a missing guard on a different direct import. Setting
    its ``sys.modules`` entry to ``None`` blocks both import statements and
    ``importlib.import_module``, which pytest uses for ``importorskip``.
    """
    for module_name in ("assimulo", *NOTEBOOK_TOOLS):
        pytest.importorskip(module_name)

    script = textwrap.dedent(f"""
        import importlib
        import sys
        import pytest

        for module_name in {('assimulo', *NOTEBOOK_TOOLS)!r}:
            importlib.import_module(module_name)
        sys.modules[{missing_module!r}] = None
        try:
            importlib.import_module({missing_module!r})
        except ModuleNotFoundError as exc:
            assert exc.name == {missing_module!r}
        else:
            raise AssertionError("optional import was not blocked")

        raise SystemExit(pytest.main([
            "tests/test_workshop_param_estimation.py",
            "tests/test_workshop_pfr_batch.py",
            "tests/test_optional_dependencies.py",
            "-m", "not assimulo", "-q", "-rs",
        ]))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed, 2 skipped" in output, output
    assert f"could not import '{missing_module}'" in output, output
