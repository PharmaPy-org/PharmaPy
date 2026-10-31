"""Compile and execute the published lomustine superstructure flowsheet guide.

Syntax checks stay in the core lane. The integration case executes the printed
code with real Assimulo collaborators across both the all-batch and hybrid
batch-reaction / continuous-MSMPR flowsheets and checks independent physical
invariants (reaction moiety balances, evaporative C7 solvent-switch enrichment,
BatchToFlowConnector discharge rate, and solid cake recovery).
"""

from collections.abc import Iterator
from pathlib import Path
import os
import re
import textwrap

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "doc/online_docs/examples/lomustine_superstructure.rst"


def _code_blocks() -> Iterator[str]:
    """Extract the guide's indented, option-free ``testcode`` directives.

    Yields
    ------
    str
        Python source from each directive, in document order.
    """
    text = GUIDE.read_text(encoding="utf-8").expandtabs()
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
def test_lomustine_guide_has_executable_blocks() -> None:
    """Keep empty extraction from silently skipping the core syntax checks."""
    assert len(BLOCKS) == 4, "Expected four executable testcode blocks in lomustine guide"


@pytest.mark.unit
@pytest.mark.parametrize(
    "source", BLOCKS, ids=[f"block-{i}" for i in range(len(BLOCKS))]
)
def test_lomustine_guide_code_compiles(source: str) -> None:
    """Require every printed block to be valid Python.

    Parameters
    ----------
    source : str
        Code extracted from one published directive.
    """
    assert source
    compile(source, f"{GUIDE.name}:testcode", "exec")


@pytest.mark.unit
@pytest.mark.parametrize(
    "working_directory", [ROOT, GUIDE.parent], ids=["root", "guide"]
)
def test_lomustine_guide_resolves_property_database_path(
    working_directory: Path,
) -> None:
    """Verify the first block resolves ``compounds_lomustine.json`` from both cwd options."""
    original_directory = Path.cwd()
    namespace = {"__name__": "__lomustine_superstructure_path_check__"}
    try:
        os.chdir(working_directory)
        exec(compile(BLOCKS[0], f"{GUIDE.name}:testcode", "exec"), namespace)
        resolved = Path(namespace["path_phys"]).resolve()
        expected = (
            ROOT / "workshop_data" / "Case_study_3" / "compounds_lomustine.json"
        ).resolve()
        assert resolved == expected
    finally:
        os.chdir(original_directory)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.slow
def test_lomustine_guide_runs_batch_and_hybrid_flowsheets() -> None:
    """Execute the printed code and check batch and hybrid physical invariants."""
    pytest.importorskip("assimulo")

    original_directory = Path.cwd()
    namespace = {"__name__": "__lomustine_superstructure_guide__"}
    try:
        os.chdir(GUIDE.parent)
        for source in BLOCKS:
            exec(compile(source, f"{GUIDE.name}:testcode", "exec"), namespace)

        sim_batch = namespace["sim_batch"]
        sim_hybrid = namespace["sim_hybrid"]

        solver_rtol = 1e-4  # [-], ODE/DAE integration relative tolerance budget
        roundoff_rtol = 1e-10  # [-], algebraic identity relative tolerance

        # 1. Verify R01 carbamoylation moiety conservation (ISO + interm == CHA + interm == c_in)
        for sim, expected_c_in in (
            (sim_batch, namespace["c_in_batch"]),
            (sim_hybrid, namespace["c_in_hybrid"]),
        ):
            r01_conc = np.asarray(sim.R01.result.mole_conc)  # [mol/L]
            iso_plus_interm = r01_conc[:, 0] + r01_conc[:, 2]  # [mol/L]
            cha_plus_interm = r01_conc[:, 1] + r01_conc[:, 2]  # [mol/L]
            np.testing.assert_allclose(
                iso_plus_interm, expected_c_in, rtol=solver_rtol, atol=0.0
            )
            np.testing.assert_allclose(
                cha_plus_interm, expected_c_in, rtol=solver_rtol, atol=0.0
            )

            # Verify R02 nitrosation produces lomustine and Evaporator enriches C7 mole fraction
            r02_conc = np.asarray(sim.R02.result.mole_conc)  # [mol/L]
            assert r02_conc[-1, 4] > 0.1  # [mol/L], positive lomustine concentration
            vap_x = np.asarray(sim.VAP01.result.x_liq)  # [-]
            assert vap_x[-1, 8] > vap_x[0, 8]  # [-], C7 mole fraction increases during THF evaporation

        # 2. Verify all-batch crystallization and filter recovery
        batch_cryst_mass = sim_batch.CR01.Outlet.Solid_1.mass  # [kg]
        batch_cake_mass = sim_batch.F01.result.mass_cake_dry[-1]  # [kg]
        assert batch_cryst_mass > 0.0
        assert batch_cake_mass == pytest.approx(
            batch_cryst_mass, rel=roundoff_rtol, abs=0.0
        )
        assert namespace["productivity_batch"] == pytest.approx(
            0.00191265, rel=1e-4
        )  # [kg/s]
        assert namespace["d10_batch"] == pytest.approx(52.0012, rel=1e-3)  # [um]

        # 3. Verify hybrid BatchToFlowConnector discharge and MSMPR -> HOLD01 -> F01 recovery
        b2f_cycle_time = namespace["b2f_cycle_time"]  # [s]
        expected_b2f_flow = (
            sim_hybrid.BatchToFlow01.Liquid_1.mass / b2f_cycle_time
        )  # [kg/s]
        assert sim_hybrid.BatchToFlow01.Outlet.mass_flow == pytest.approx(
            expected_b2f_flow, rel=roundoff_rtol, abs=0.0
        )
        hybrid_held_solids = sim_hybrid.HOLD01.Solid_1.mass  # [kg]
        hybrid_cake_mass = sim_hybrid.F01.result.mass_cake_dry[-1]  # [kg]
        assert hybrid_held_solids > 0.0
        assert hybrid_cake_mass == pytest.approx(
            hybrid_held_solids, rel=roundoff_rtol, abs=0.0
        )
        assert namespace["productivity_hybrid"] == pytest.approx(
            0.00141187, rel=1e-4
        )  # [kg/s]
        assert namespace["d10_hybrid"] == pytest.approx(46.9849, rel=1e-3)  # [um]
    finally:
        os.chdir(original_directory)
