"""Compile and execute the published step-by-step flowsheet guide.

Syntax checks stay in the core lane. The integration case executes the printed
code with real Assimulo collaborators and the guide's nominal teaching inputs.
"""

from collections.abc import Iterator
from pathlib import Path
import os
import re
import textwrap

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "doc/online_docs/examples/PharmaPy_guide.rst"


def _code_blocks() -> Iterator[str]:
    """Extract the guide's indented, option-free ``testcode`` directives.

    Yields
    ------
    str
        Python source from each directive, in document order. Tabs are expanded
        before comparing indentation, following reStructuredText indentation.
    """
    text = GUIDE.read_text().expandtabs()
    for match in re.finditer(r"^( *)\.\. testcode::\s*\n", text, re.MULTILINE):
        indentation = len(match.group(1))
        lines = []
        for line in text[match.end():].splitlines():
            if line.strip() and len(line) - len(line.lstrip()) <= indentation:
                break
            lines.append(line)
        yield textwrap.dedent("\n".join(lines)).strip()


BLOCKS = tuple(_code_blocks())


@pytest.mark.unit
def test_guide_has_executable_blocks() -> None:
    """Keep empty extraction from silently skipping the core syntax checks."""
    assert BLOCKS, "No executable guide blocks were extracted"


@pytest.mark.unit
@pytest.mark.parametrize("source", BLOCKS, ids=[f"block-{i}" for i in range(len(BLOCKS))])
def test_guide_code_compiles(source: str) -> None:
    """Require every printed block to be valid Python.

    Parameters
    ----------
    source : str
        Code extracted from one published directive.
    """
    assert source
    compile(source, f"{GUIDE.name}:testcode", "exec")


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("working_directory", [ROOT, GUIDE.parent], ids=["root", "guide"])
def test_guide_runs_complete_flowsheet(working_directory: Path) -> None:
    """Execute the printed code and check the nominal process endpoints.

    Parameters
    ----------
    working_directory : pathlib.Path
        Supported starting directory within the checkout.
    """
    pytest.importorskip("assimulo")
    import matplotlib.pyplot as plt

    original_directory = Path.cwd()
    namespace = {"__name__": "__teaching_guide__"}
    try:
        os.chdir(working_directory)
        for source in BLOCKS:
            exec(compile(source, f"{GUIDE.name}:testcode", "exec"), namespace)
        sim = namespace["sim"]
        # The documented nominal process collects for two hours and cools
        # for four hours to 278.15 K. Endpoint tolerance permits round-off.
        roundoff = 1e-10  # [-], well below default solver relative tolerance
        reactor_runtime = 7200.0  # [s], two-hour teaching campaign
        crystal_runtime = 14400.0  # [s], four-hour cooling program
        final_temperature = 278.15  # [K], documented program endpoint
        assert sim.R01.result.time[-1] == pytest.approx(
            reactor_runtime, rel=roundoff, abs=0)
        assert sim.CR01.result.time[-1] == pytest.approx(
            crystal_runtime, rel=roundoff, abs=0)
        assert sim.CR01.result.temp[-1] == pytest.approx(
            final_temperature, rel=roundoff, abs=0)
        assert np.isfinite(sim.CR01.result.mu_n).all()
        crystal_mass = sim.CR01.Outlet.Solid_1.mass  # [kg], filter feed solids
        cake_mass = sim.F01.result.mass_cake_dry[-1]  # [kg], recovered dry solids
        assert crystal_mass > 0
        assert cake_mass == pytest.approx(crystal_mass, rel=roundoff, abs=0)
        np.testing.assert_allclose(
            sim.R01.Inlet.mole_conc[:4], [0.33, 0.33, 0, 0],
            rtol=roundoff, atol=0)  # [mol/L], named A, B, C, D feed
        assert sim.R01.name_species == ["A", "B", "C", "D", "solvent"]
    finally:
        os.chdir(original_directory)
        plt.close("all")
