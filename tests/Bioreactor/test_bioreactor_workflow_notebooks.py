"""The example notebooks only simulate and export reproducible trajectories."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples/bioreactors"
NOTEBOOKS = tuple(EXAMPLES / name / "workflow.ipynb" for name in (
    "batch_ecoli_dfba", "generic_batch", "generic_fed_batch"))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_workflow_notebook_executes_from_repository_root(monkeypatch, tmp_path, notebook):
    document = json.loads(notebook.read_text())
    namespace = {"__name__": "__notebook__"}
    monkeypatch.chdir(ROOT)
    # The notebooks need only the forward inputs, even when auxiliary files are absent.
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("case.json", "mechanism.json", "thermo.json"):
        (inputs / name).write_bytes((notebook.parent / "inputs" / name).read_bytes())
    try:
        for step, cell in enumerate((c for c in document["cells"] if c["cell_type"] == "code"), 1):
            exec(compile("".join(cell["source"]), f"{notebook}:step-{step}", "exec"), namespace)
            if step == 1:
                namespace.update(input_dir=inputs, thermo_path=inputs / "thermo.json", EXPORT_DIR=tmp_path)
        rows = namespace["rows"]
        saved = np.genfromtxt(tmp_path / "trajectories.csv", delimiter=",", names=True)
        assert len(saved) == sum(len(r.time) for r in namespace['histories'])
        for name in saved.dtype.names:
            np.testing.assert_allclose(saved[name], [row[name] for row in rows], rtol=1e-14)
            assert np.isfinite(saved[name]).all()
        for extension in ('png', 'svg'):
            assert (tmp_path / f"trajectories.{extension}").stat().st_size > 1000
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            'inputs', 'trajectories.csv', 'trajectories.png', 'trajectories.svg']
        for segment in namespace['histories']:
            assert np.min(segment.mass_j_liquid0) >= -1e-10
            assert np.min(segment.vessel_vol) > 0.
    finally:
        plt.close('all')


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_workflow_notebook_uses_library_directly(notebook):
    document = json.loads(notebook.read_text())
    cells = [c for c in document['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 6
    source = '\n'.join(''.join(c['source']) for c in cells)
    assert 'build_bioreactor(' in source and 'assembly.solve(' in source
    for forbidden in ('run_design_study', 'estimate_bioreactor_parameters', 'REFERENCE_INPUT_SHA256',
                      'design.json', 'estimation.json', 'regression_baseline', 'load_runner', 'scipy.optimize'):
        assert forbidden not in source
    for step, cell in enumerate(cells, 1):
        text = ''.join(cell['source'])
        assert text.startswith(f'# STEP {step}:')
        assert sum(line.startswith('# ') for line in text.splitlines()[1:6]) >= 2
        compile(text, str(notebook), 'exec')


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
