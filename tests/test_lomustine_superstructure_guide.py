"""Compile and execute the published lomustine superstructure flowsheet guide.

Syntax, path-resolution, optional-dependency independence, and solubility contract
checks run in the core lane. The integration case executes the printed code with
real Assimulo collaborators across both the all-batch and hybrid
batch-reaction / continuous-MSMPR flowsheets and checks independent physical
invariants (reaction moiety balances across ``R01`` and ``R02``, upper bounds on
recovered lomustine from limiting reactant inventory, evaporative ``C7``
solvent-switch enrichment, ``BatchToFlowConnector`` mass/volumetric discharge and
actual MSMPR residence time, and solid cake recovery) alongside deterministic
regression baselines.
"""

from collections.abc import Iterator
import json
from pathlib import Path
import os
import re
import subprocess
import sys
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
    """Verify the first block resolves ``compounds_lomustine.json`` from both cwd options.

    Parameters
    ----------
    working_directory : pathlib.Path
        Directory from which the documentation snippet is executed (either the
        repository root or ``doc/online_docs/examples``).
    """
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

        db = json.loads(resolved.read_text(encoding="utf-8"))
        assert db["SUB1"]["formula"] == "C7H14ClNO2"
        assert db["SUB1"]["mw"] == pytest.approx(179.643, rel=1e-9)  # [g/mol]
        assert db["C7"]["cas"] == "142-82-5"
    finally:
        os.chdir(original_directory)


@pytest.mark.unit
def test_lomustine_guide_blocks_0_and_1_run_without_assimulo() -> None:
    """Verify blocks 0 and 1 execute in an isolated subprocess when Assimulo is absent."""
    probe = textwrap.dedent(
        f"""
        import builtins
        import sys

        real_import = builtins.__import__

        def _guard_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "assimulo" or name.startswith("assimulo."):
                raise ModuleNotFoundError("Blocked optional assimulo import")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _guard_import
        namespace = {{"__name__": "__no_assimulo_probe__"}}
        exec({BLOCKS[0]!r}, namespace)
        exec({BLOCKS[1]!r}, namespace)
        sol = namespace["lomustine_solubility"](298.15)
        assert sol > 0.0
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(GUIDE.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_lomustine_solubility_input_validation_and_precedence() -> None:
    """Check 1D/2D composition forms, option precedence, and invalid-input errors."""
    original_directory = Path.cwd()
    namespace = {"__name__": "__lomustine_solubility_contract__"}
    try:
        os.chdir(GUIDE.parent)
        exec(compile(BLOCKS[0], f"{GUIDE.name}:testcode", "exec"), namespace)
        exec(compile(BLOCKS[1], f"{GUIDE.name}:testcode", "exec"), namespace)
        solub_fn = namespace["lomustine_solubility"]
        mw = namespace["SPECIES_MW"]  # [g/mol]

        temp_scalar = 295.0  # [K]
        base_only = solub_fn(temp_scalar)  # [kg/m**3], default x_thf = 0
        mole_frac_1d = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.25, 0.70])  # [-]
        conc_1d = mole_frac_1d * mw  # [kg/m**3], proportional mass concentrations

        from_mole_frac = solub_fn(temp_scalar, mole_frac=mole_frac_1d)  # [kg/m**3]
        from_conc = solub_fn(temp_scalar, conc=conc_1d)  # [kg/m**3]
        assert from_conc == pytest.approx(from_mole_frac, rel=1e-12)
        assert from_conc == pytest.approx(base_only * (1.0 + 6.5 * 0.25), rel=1e-12)

        # Verify conc takes precedence over mole_frac when both are supplied
        other_mole_frac = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.20])  # [-]
        assert solub_fn(
            temp_scalar, conc=conc_1d, mole_frac=other_mole_frac
        ) == pytest.approx(from_conc, rel=1e-12)

        # Verify 2D batched evaluation
        temps_2d = np.array([280.0, 295.0])  # [K]
        conc_2d = np.vstack([conc_1d, conc_1d * 2.0])  # [kg/m**3]
        sol_2d = solub_fn(temps_2d, conc=conc_2d)  # [kg/m**3]
        assert sol_2d.shape == (2,)
        assert sol_2d[1] == pytest.approx(from_conc, rel=1e-12)

        # Invalid inputs must raise ValueError
        with pytest.raises(ValueError, match="finite"):
            solub_fn(np.nan)
        with pytest.raises(ValueError, match="shape"):
            solub_fn(temp_scalar, conc=np.ones(5))
        with pytest.raises(ValueError, match="non-negative"):
            bad_conc = conc_1d.copy()
            bad_conc[7] = -1.0
            solub_fn(temp_scalar, conc=bad_conc)
        with pytest.raises(ValueError, match="positive total"):
            solub_fn(temp_scalar, conc=np.zeros(9))
        with pytest.raises(ValueError, match="shape"):
            solub_fn(temp_scalar, mole_frac=np.ones((2, 3, 9)))
        with pytest.raises(ValueError, match="non-negative"):
            bad_x = mole_frac_1d.copy()
            bad_x[8] = -0.1
            solub_fn(temp_scalar, mole_frac=bad_x)
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
        lom_mw_kg_per_mol = 0.2337  # [kg/mol], lomustine molecular weight (233.7 g/mol)
        liters_per_m3 = 1000.0  # [L/m**3], exact volume conversion

        # 1. Verify R01 and R02 moiety conservation and Evaporator solvent switch
        for sim, expected_c_in, init_vol in (
            (sim_batch, namespace["c_in_batch"], namespace["vol_batch"]),
            (sim_hybrid, namespace["c_in_hybrid"], namespace["vol_hybrid"]),
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

            # In R02: interm + TBN -> lom + TBA and ISO + TBA -> SUB1
            # Thus CHA + interm + lom and ISO + interm + lom + SUB1 are invariant in R02.
            r02_conc = np.asarray(sim.R02.result.mole_conc)  # [mol/L]
            r02_cha_moiety = r02_conc[:, 1] + r02_conc[:, 2] + r02_conc[:, 4]  # [mol/L]
            r02_iso_moiety = (
                r02_conc[:, 0] + r02_conc[:, 2] + r02_conc[:, 4] + r02_conc[:, 6]
            )  # [mol/L]
            np.testing.assert_allclose(
                r02_cha_moiety, r02_cha_moiety[0], rtol=solver_rtol, atol=0.0
            )
            np.testing.assert_allclose(
                r02_iso_moiety, r02_iso_moiety[0], rtol=solver_rtol, atol=0.0
            )

            # Lomustine formed in R02 must equal the drop in intermediate concentration
            lom_formed = r02_conc[-1, 4] - r02_conc[0, 4]  # [mol/L]
            interm_consumed = r02_conc[0, 2] - r02_conc[-1, 2]  # [mol/L]
            assert lom_formed > 0.1  # [mol/L]
            assert lom_formed == pytest.approx(interm_consumed, rel=solver_rtol)

            vap_x = np.asarray(sim.VAP01.result.x_liq)  # [-]
            assert vap_x[-1, 8] > vap_x[0, 8]  # [-], C7 mole fraction increases during THF evaporation

            # Total lomustine moles recovered in F01 cake cannot exceed initial CHA moles (35 mol)
            initial_cha_moles = expected_c_in * init_vol * liters_per_m3  # [mol]
            cake_lom_moles = sim.F01.result.mass_cake_dry[-1] / lom_mw_kg_per_mol  # [mol]
            assert 0.0 < cake_lom_moles < initial_cha_moles

        # 2. Verify all-batch crystallization and filter recovery
        # Deterministic regression baselines at combo "0" of optimal_points_2.json
        # (SECQUOIA/pd_ising; Park & Bernal Neira, 2026) evaluated on PharmaPy master:
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

        # 3. Verify hybrid BatchToFlowConnector discharge, actual MSMPR residence time, and recovery
        # Deterministic regression baselines at combo "1" of optimal_points_2.json
        # (SECQUOIA/pd_ising; Park & Bernal Neira, 2026) evaluated on PharmaPy master:
        b2f_cycle_time = namespace["b2f_cycle_time"]  # [s]
        expected_b2f_mass_flow = (
            sim_hybrid.BatchToFlow01.Liquid_1.mass / b2f_cycle_time
        )  # [kg/s]
        expected_b2f_vol_flow = (
            sim_hybrid.BatchToFlow01.Liquid_1.vol / b2f_cycle_time
        )  # [m**3/s]
        assert sim_hybrid.BatchToFlow01.Outlet.mass_flow == pytest.approx(
            expected_b2f_mass_flow, rel=roundoff_rtol, abs=0.0
        )
        assert sim_hybrid.BatchToFlow01.Outlet.vol_flow == pytest.approx(
            expected_b2f_vol_flow, rel=roundoff_rtol, abs=0.0
        )
        assert namespace["tau_actual_cr01"] == pytest.approx(
            namespace["vol_cryst_hybrid"] / expected_b2f_vol_flow,
            rel=roundoff_rtol,
            abs=0.0,
        )  # [s]

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
