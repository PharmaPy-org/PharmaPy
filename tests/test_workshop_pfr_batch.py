"""Execute the PFR/batch teaching notebook with real CVode collaborators.

Two quick cases cover nominal simulation/costing from both launch directories.
One complete run retains both original optimization budgets (~5 minutes locally).
Historical cost values are preserved; the old optimized design is not certified.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("assimulo")
pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("jupyter_client")

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

pytestmark = [pytest.mark.assimulo, pytest.mark.integration, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "doc/online_docs/examples/PFR_Batch_solved.ipynb"
GOLDEN = ROOT / "tests/fixtures/pfr_batch_2023.json"
# The original continuation took 543 s; allow about twice that duration per cell.
CELL_TIMEOUT = 1200  # [s], execution allowance, not an optimizer stopping rule
# Costing and solid handoff are algebraic, so permit round-off, not solver drift.
ROUND_OFF = 1e-10  # [-], conservative allowance for floating-point conversions


def _execute(notebook, working_directory: Path) -> None:
    """Execute the supplied cells in a fresh kernel and release it.

    Parameters
    ----------
    notebook : nbformat.NotebookNode
        Notebook whose cells and outputs are updated in place.
    working_directory : pathlib.Path
        Initial kernel directory inside the repository checkout.
    """
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=CELL_TIMEOUT, allow_errors=False,
        resources={"metadata": {"path": str(working_directory)}},
    )
    client.execute(cleanup_kc=True)
    assert not manager.has_kernel


@pytest.mark.parametrize("working_directory", [ROOT, NOTEBOOK.parent], ids=["root", "notebook"])
def test_pfr_notebook_nominal(working_directory: Path) -> None:
    """Check the printed nominal flowsheet and initial costing calculation.

    Parameters
    ----------
    working_directory : pathlib.Path
        Supported notebook launch directory.

    Notes
    -----
    Stop before optimization for this quick check. The complete case below
    executes every cell with the original optimizer budgets.
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    optimization_cells = [
        index for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and cell.source.startswith("from scipy.optimize import minimize")
    ]
    assert len(optimization_cells) == 1
    notebook.cells = notebook.cells[:optimization_cells[0]]
    probe = nbformat.v4.new_code_cell(
        "from IPython.display import JSON, display\n"
        "display(JSON({'species': flst.R01.name_species, "
        "'costs': di_callback['cost/batch'], "
        "'crystal_mass': float(flst.CR01.Outlet.Solid_1.mass), "
        "'cake_mass': float(flst.F01.result.mass_cake_dry[-1]), "
        "'final_temp': float(flst.CR01.result.temp[-1]), "
        "'feed_volume': float(flst.R01.Inlet.vol_flow * runtime_reactor)}))"
    )
    notebook.cells.append(probe)
    _execute(notebook, working_directory)
    actual = probe.outputs[0].data["application/json"]
    historical = json.loads(GOLDEN.read_text())
    assert actual["species"] == historical["species"]
    assert list(actual["costs"]) == historical["cost_columns"]
    np.testing.assert_allclose(
        list(actual["costs"].values()), historical["cost_per_batch"],
        rtol=ROUND_OFF, atol=0,
    )
    # Independent feed costing: 60 L at 0.33 mol/L gives 19.8 mol per reactant;
    # A is 100 g/mol at 10 USD/kg, B is 50 g/mol at 12 USD/kg.
    expected_reactant_costs = [19.8, 11.88]  # [USD/batch], derived above
    np.testing.assert_allclose(
        [actual["costs"]["mass_A"], actual["costs"]["mass_B"]],
        expected_reactant_costs, rtol=ROUND_OFF, atol=0,
    )
    assert actual["crystal_mass"] > 0
    assert actual["cake_mass"] == pytest.approx(actual["crystal_mass"], rel=ROUND_OFF, abs=0)
    expected_feed_volume = 0.04  # [m**3], 0.010 m**3 / 1800 s * 7200 s
    final_temperature = 278.15  # [K], the nominal cooling program endpoint
    assert actual["feed_volume"] == pytest.approx(expected_feed_volume, rel=ROUND_OFF, abs=0)
    assert actual["final_temp"] == pytest.approx(final_temperature, rel=ROUND_OFF, abs=0)


def test_pfr_notebook_complete() -> None:
    """Execute every cell and preserve honest reporting of a bounded search.

    Notes
    -----
    Issue #172 owns reproduction of the historical optimized design. Neither
    the original stored result nor current execution establishes convergence;
    this check enforces finite outputs and truthful status, not feasibility.
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    probe = nbformat.v4.new_code_cell(
        "from IPython.display import JSON, display\n"
        "display(JSON({'success': bool(result_two.success), "
        "'nfev': int(result_two.nfev), 'x': result_two.x.tolist(), "
        "'bounds': bounds, 'objective': float(result_two.fun), "
        "'columns': x_summary.columns.tolist(), "
        "'mean_undefined_initially': bool(np.isnan(mean_size_profile[0])), "
        "'mean_defined_with_particles': bool(np.isfinite(mean_size_profile[moms[:, 0] > 0]).all()), "
        "'has_particles': bool((moms[:, 0] > 0).any())}))"
    )
    notebook.cells.append(probe)
    _execute(notebook, NOTEBOOK.parent)
    actual = probe.outputs[0].data["application/json"]
    assert actual["columns"] == ["initial", "final iterate"]
    continuation_budget = 400  # [-], unchanged 2023 tutorial function-evaluation budget
    assert 0 < actual["nfev"] <= continuation_budget
    decision = np.asarray(actual["x"])  # [s, K, K, K, s, Pa]
    bounds = np.asarray(actual["bounds"])  # [s, K, K, K, s, Pa], lower/upper columns
    assert np.isfinite(decision).all() and np.isfinite(actual["objective"])
    assert decision.shape == (6,)
    assert np.all(decision >= bounds[:, 0]) and np.all(decision <= bounds[:, 1])
    assert actual["mean_undefined_initially"]
    assert actual["has_particles"] and actual["mean_defined_with_particles"]
    streams = "\n".join(
        output.text for cell in notebook.cells for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert f"Optimizer success: {actual['success']}" in streams
