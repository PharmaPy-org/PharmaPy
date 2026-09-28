"""Execute the parameter-estimation workshop with real CVode and historical data.

Each case runs a fresh kernel in the optional Assimulo environment (~10 s).
The 2023 parameter table is an immutable reference, not regenerated test data.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("assimulo")

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from scipy.linalg import expm

pytestmark = [pytest.mark.assimulo, pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "doc/online_docs/examples/param_estimation_solved.ipynb"
GOLDEN = ROOT / "tests/fixtures/param_estimation_2023.json"
# CVode's default relative tolerance; this checks compatibility of the fitted
# parameters, not their statistical uncertainty or cross-version certification.
FIT_RTOL = 1e-6  # [-]
# The historical printed table rounds each column to six decimal places.
PRINT_ATOL = 0.5e-6  # [-] for ln(A / (1/s)); [K] for Ea/R
# Ten default CVode absolute tolerances allow accumulated integration error
# across the coupled four-species system; all concentrations start at <= 1 mol/L.
CONCENTRATION_ATOL = 1e-5  # [mol/L]
CELL_TIMEOUT = 180  # [s], generous CI allowance for a cell normally taking < 3 s


@pytest.mark.parametrize("working_directory", [ROOT, NOTEBOOK.parent], ids=["root", "notebook"])
def test_parameter_estimation_notebook(working_directory: Path) -> None:
    """Run every cell and verify independent balances and historical fit values.

    Parameters
    ----------
    working_directory : pathlib.Path
        Supported notebook launch directory within the repository checkout.

    Notes
    -----
    Test-only probe cells capture full-precision values without changing the
    tracked notebook. No output images, timings, or iteration counts are golden.
    The zero-inventory and missing-Utility warnings remain visible.
    https://github.com/PharmaPy-org/PharmaPy/issues/172 tracks their review
    before treating this teaching example as a physical validation.
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nominal_cells = [
        index for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and cell.source.startswith("time, states, sens = reactor.solve_unit(runtime=120")
    ]
    assert len(nominal_cells) == 1
    nominal_probe = nbformat.v4.new_code_cell(
        "import numpy as np\nfrom IPython.display import JSON, display\n"
        "display(JSON({'time': np.asarray(time).tolist(), 'states': np.asarray(states).tolist()}))"
    )
    notebook.cells.insert(nominal_cells[0] + 1, nominal_probe)
    final_probe = nbformat.v4.new_code_cell(
        "display(JSON({'reactions': optim_df.index.tolist(), "
        "'columns': optim_df.columns.tolist(), 'fit': optim_df.values.tolist()}))"
    )
    notebook.cells.append(final_probe)

    manager = KernelManager(kernel_name="python3")
    # A globally registered python3 kernel may belong to another environment.
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=CELL_TIMEOUT, allow_errors=False,
        resources={"metadata": {"path": str(working_directory)}},
    )
    client.execute(cleanup_kc=True)
    assert not manager.has_kernel

    nominal = nominal_probe.outputs[0].data["application/json"]
    time = np.asarray(nominal["time"])  # [s]
    states = np.asarray(nominal["states"])  # [mol/L], A, B, C, D
    assert time.size > 1
    np.testing.assert_allclose(time[[0, -1]], [0, 120], rtol=0, atol=0)
    assert states.shape == (time.size, 4)
    # Independent first-order material balance for A->B, A->C, B->C, B->D,
    # each with k=0.1/s and initially pure A at 1 mol/L. exp(M*t)c0 is exact.
    rate = 0.1  # [1/s], the notebook's explicitly chosen demonstration kinetics
    generator = rate * np.array([
        [-2, 0, 0, 0], [1, -2, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]
    ])  # [1/s], columns consume a species and produce its reaction products
    initial = np.array([1, 0, 0, 0])  # [mol/L], the notebook's initial condition
    expected = np.array([expm(generator * point) @ initial for point in time])  # [mol/L]
    np.testing.assert_allclose(states, expected, rtol=0, atol=CONCENTRATION_ATOL)
    np.testing.assert_allclose(states.sum(axis=1), 1, rtol=0, atol=CONCENTRATION_ATOL)

    actual = final_probe.outputs[0].data["application/json"]
    historical = json.loads(GOLDEN.read_text())
    assert actual["reactions"] == historical["reactions"]
    assert actual["columns"] == historical["columns"]
    np.testing.assert_allclose(
        actual["fit"], historical["fit"], rtol=FIT_RTOL, atol=PRINT_ATOL,
    )
