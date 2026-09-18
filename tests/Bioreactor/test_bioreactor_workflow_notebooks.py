"""Execution and direct-workflow checks for the two WP3 workflow notebooks."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples/bioreactors"
NOTEBOOKS = (
    EXAMPLES / "batch_ecoli_dfba/workflow.ipynb",
    EXAMPLES / "fed_batch_cho/workflow.ipynb",
)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=("ecoli", "cho"))
def test_workflow_notebook_executes_from_repository_root(monkeypatch, notebook):
    """Execute code cells without requiring Jupyter as a runtime dependency."""
    document = json.loads(notebook.read_text())
    namespace = {"__name__": "__notebook__"}
    monkeypatch.chdir(ROOT)
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{notebook.name}:cell-{index}", "exec"), namespace)
    assert namespace["workflow_result"]["status"] == "PASS"
    assert namespace["workflow_result"]["trajectory_agrees"] is True
    assert namespace["workflow_result"]["metrics_agree"] is True


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=("ecoli", "cho"))
def test_workflow_notebook_uses_library_directly(notebook):
    """Expose editable steps without copying solvers or importing example runners."""
    document = json.loads(notebook.read_text())
    code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "build_bioreactor(" in source
    assert "run_design_study(" in source
    assert "assembly.solve(" in source
    assert "reference_inputs_match" in source
    assert "WRITE_ARTIFACTS = False" in source
    assert "importlib" not in source
    assert "load_runner" not in source
    assert "examples.bioreactors" not in source
    assert "scipy.optimize" not in source
    for cell in code_cells:
        text = "".join(cell["source"])
        lines = text.splitlines()
        assert lines[0].startswith("# STEP ")
        assert sum(line.startswith("# ") for line in lines[1:6]) >= 2
        compile(text, str(notebook), "exec")


def test_runtime_does_not_depend_on_notebooks_or_example_runners():
    """Deleting presentation clients must not affect installed capabilities."""
    production = "\n".join(
        path.read_text() for path in (ROOT / "PharmaPy").rglob("*.py")
    )
    assert ".ipynb" not in production
    assert "run_ecoli" not in production
    assert "run_cho" not in production


def test_notebook_tree_contains_no_transient_artifacts():
    forbidden_names = {".ipynb_checkpoints", "notebook_outputs"}
    offenders = [
        path for path in EXAMPLES.rglob("*")
        if path.name in forbidden_names or path.suffix in {".tmp", ".bak"}
    ]
    assert offenders == []


@pytest.mark.parametrize("change, expected_error", [
    ("observed_drift", None),
    ("concentration_regression", "CHO trajectory: ASP"),
    ("metric_regression", "CHO summary: relative_final_product_error"),
    ("nonfinite", "nonfinite generated values"),
    ("time_grid", "CHO trajectory: time_day"),
    ("design_selection", "CHO design selection changed"),
    ("acceptance_failure", "CHO simulation acceptance failed"),
])
def test_cho_notebook_regression_gate(change, expected_error):
    """Allow measured numerical drift while rejecting meaningful output changes."""
    import numpy as np

    folder = EXAMPLES / "fed_batch_cho"
    outputs = folder / "outputs"
    trajectory = np.genfromtxt(outputs / "trajectories.csv", delimiter=",", names=True)
    rows = [{name: float(row[name]) for name in trajectory.dtype.names}
            for row in trajectory]
    summary = json.loads((outputs / "summary.json").read_text())
    design = json.loads((outputs / "experiment_design.json").read_text())
    if change == "observed_drift":
        # Exercise small relative drift without assuming historical concentrations.
        asp_index = int(np.argmin(abs(trajectory["time_day"] - 6.1)))
        nh3_index = int(np.argmin(abs(trajectory["time_day"] - 5.3)))
        rows[asp_index]["ASP"] *= 1.0 - 2.4e-5
        rows[nh3_index]["NH3"] *= 1.0 - 1.2e-5
        summary["relative_final_product_error"] += 5.2122857251883994e-8
    elif change == "concentration_regression":
        rows[0]["ASP"] *= 1.01
    elif change == "metric_regression":
        summary["relative_final_product_error"] += 1e-4
    elif change == "nonfinite":
        rows[0]["ASP"] = np.nan
    elif change == "time_grid":
        rows[0]["time_day"] += 1e-8
    elif change == "design_selection":
        design["selected_id"] = "unexpected-condition"
    elif change == "acceptance_failure":
        summary["status"] = "FAIL"
    namespace = {"np": np, "json": json, "reference_dir": folder, "check_regression": True,
                 "simulation": {"rows": rows, "summary": summary}, "design": design}
    document = json.loads((folder / "workflow.ipynb").read_text())
    source = "".join(next(cell for cell in document["cells"]
                          if "canonical-regression" in cell.get("metadata", {}).get("tags", []))["source"])
    if expected_error is None:
        exec(compile(source, "cho-regression-gate", "exec"), namespace)
        assert namespace["workflow_result"]["status"] == "PASS"
    else:
        with pytest.raises(AssertionError, match=expected_error):
            exec(compile(source, "cho-regression-gate", "exec"), namespace)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=("ecoli", "cho"))
def test_custom_notebook_configuration_skips_original_regression(notebook):
    """Custom inputs must not be assessed against the distributed model baseline."""
    import numpy as np
    document = json.loads(notebook.read_text())
    namespace = {"np": np, "json": json, "check_regression": False,
                 "simulation": {"summary": {"status": "NOT_CHECKED", "native_unit": "custom"}},
                 "design": {"status": "PASS", "capability_status": "PASS",
                            "empirical_benefit_status": "NOT_DEMONSTRATED"}}
    source = "".join(next(cell for cell in document["cells"]
                          if "canonical-regression" in cell.get("metadata", {}).get("tags", []))["source"])
    exec(compile(source, "custom-regression-gate", "exec"), namespace)
    assert namespace["workflow_result"]["regression_status"] == "NOT_CHECKED"
    assert namespace["workflow_result"]["trajectory_agrees"] is None


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=("ecoli", "cho"))
def test_notebook_detects_edited_input_files(monkeypatch, tmp_path, notebook):
    """Editing a JSON file must not redefine what counts as the original case."""
    import shutil
    document = json.loads(notebook.read_text())
    cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    namespace = {"__name__": "__notebook__"}
    monkeypatch.chdir(ROOT)
    exec(compile("".join(cells[0]["source"]), "setup", "exec"), namespace)
    copied = tmp_path / "inputs"
    shutil.copytree(notebook.parent / "inputs", copied)
    namespace["input_dir"] = copied
    namespace["thermo_path"] = copied / "thermo.json"
    input_source = "".join(cells[1]["source"])
    validation_source = "".join(next(
        cell["source"] for cell in cells
        if "REFERENCE_INPUT_SHA256" in "".join(cell["source"])
    ))
    start = validation_source.index("configured_documents =")
    ends = [validation_source.index(marker) for marker in ("baseline =", "final_vcd =")
            if marker in validation_source]
    provenance_source = validation_source[start:min(ends)]
    exec(compile(input_source, "inputs", "exec"), namespace)
    exec(compile(provenance_source, "provenance", "exec"), namespace)
    assert namespace["check_regression"] is True
    case = json.loads((copied / "case.json").read_text())
    case["operation"]["runtime"]["value"] *= 0.5
    (copied / "case.json").write_text(json.dumps(case))
    exec(compile(input_source, "edited-inputs", "exec"), namespace)
    exec(compile(provenance_source, "edited-provenance", "exec"), namespace)
    assert namespace["check_regression"] is False
