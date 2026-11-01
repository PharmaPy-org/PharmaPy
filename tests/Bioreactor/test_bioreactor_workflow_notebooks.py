"""The example notebooks only simulate and export reproducible trajectories."""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples/bioreactors"
NOTEBOOKS = tuple(EXAMPLES / name / "workflow.ipynb" for name in (
    "batch_ecoli_dfba", "generic_batch", "generic_fed_batch", "generic_chemostat")) + (
        EXAMPLES / 'Doyle_Exercise_4_15/workbook.ipynb',)


# Real fixture construction and independent qualification helpers.
def _check_notebook(tmp_path, notebook, input_source=None, sample=False):
    """Execute the notebook and check exported values in an isolated process.

    Parameters
    ----------
    tmp_path : Path
        Disposable input/export directory.
    notebook : Path
        Notebook source; trajectory units are declared by its column names.
    input_source : Path, optional
        Alternate inputs folder, e.g. the direct-kinetics chemostat.
    sample : bool, optional
        Add a 10% well-mixed sample at half the declared duration to expose
        event segmentation and biomass representation in the exported table.
    """
    document = json.loads(notebook.read_text())
    namespace = {"__name__": "__notebook__"}
    # The notebooks need only the forward inputs, even when auxiliary files are absent.
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    input_source = notebook.parent / "inputs" if input_source is None else input_source
    for name in ("mechanism.json", "thermo.json"):
        (inputs / name).write_bytes((input_source / name).read_bytes())
    if sample:
        model_path = inputs / 'mechanism.json'
        model = json.loads(model_path.read_text())
        sample_time = model['operation']['runtime']['value'] / 2.  # [h], midpoint of this fixture
        model['recipes'][model['operation']['recipe']['name']].append(
            {'event_type': 'sample', 'time': sample_time, 'volume_l': .1})  # [h, L], 10% of 1 L
        model_path.write_text(json.dumps(model))
    try:
        for step, cell in enumerate((c for c in document["cells"] if c["cell_type"] == "code"), 1):
            exec(compile("".join(cell["source"]), f"{notebook}:step-{step}", "exec"), namespace)
            if step == 1:
                namespace.update(input_dir=inputs, thermo_path=inputs / "thermo.json", EXPORT_DIR=tmp_path)
        rows = namespace["rows"]
        saved = np.genfromtxt(tmp_path / "trajectories.csv", delimiter=",", names=True)
        assert len(saved) == sum(len(r.time) for r in namespace['histories'])
        if notebook.parent.name in ('generic_batch', 'generic_chemostat'):
            assert saved.dtype.names == ('time_h', 'nutrient_mmol', 'biomass_gdw',
                'liquid_volume_l', 'nutrient_mmol_l', 'biomass_gdw_l',
                'liquid_plus_biomass_kg', 'segment_index')
        for name in saved.dtype.names:
            np.testing.assert_allclose(saved[name], [row[name] for row in rows], rtol=1e-14)
            assert np.isfinite(saved[name]).all()
        if sample:
            # One representable time increment [h] permits conversion roundoff only.
            event_rows = saved[np.isclose(saved['time_h'], sample_time,
                                          rtol=0., atol=np.spacing(sample_time))]
            assert len(event_rows) == 2
            # A 0.1 L sample removes 10% of a 1 L vessel without changing concentrations.
            # The declared 1e-9 relative solver target is 1e-9 L for this 1 L fixture.
            np.testing.assert_allclose(event_rows['liquid_volume_l'], [1., .9], rtol=0., atol=1e-9)
            # Match the configured solver's relative target for concentration continuity.
            for field in ('biomass_gdw_l', 'nutrient_mmol_l'):
                assert event_rows[field][1] == pytest.approx(event_rows[field][0], rel=1e-9)
            assert event_rows['segment_index'].tolist() == [0, 1]
        if namespace['config']['mechanism'] == 'configured-kinetics' and 'liquid_plus_biomass_kg' in saved.dtype.names:
            # Direct biomass already belongs to phase mass; a second addition would double count it.
            expected_mass = np.concatenate([np.asarray(h.mass_j_liquid0).sum(axis=1)
                                            for h in namespace['histories']])  # [kg]
            np.testing.assert_allclose(saved['liquid_plus_biomass_kg'], expected_mass,
                                       rtol=0., atol=1e-12)  # [kg], 1e-12 relative to this 1 kg fixture
        for extension in ('png', 'svg'):
            assert (tmp_path / f"trajectories.{extension}").stat().st_size > 1000
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            'inputs', 'trajectories.csv', 'trajectories.png', 'trajectories.svg']
        for segment in namespace['histories']:
            assert np.min(segment.mass_j_liquid0) >= -1e-10
            assert np.min(segment.vessel_vol) > 0.
    finally:
        plt.close('all')
# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_workflow_notebook_executes_from_repository_root(tmp_path, notebook):
    """Execute native example workflow cells in an isolated process.

    Parameters
    ----------
    tmp_path : Path
        Disposable output directory.
    notebook : Path
        Notebook source to verify.
    """
    result = subprocess.run([sys.executable, '-c',
        "import runpy, pathlib, sys; "
        "runpy.run_path(sys.argv[1])['_check_notebook'](pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))",
        str(Path(__file__).resolve()), str(tmp_path), str(notebook)],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


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


@pytest.mark.integration
@pytest.mark.parametrize('variant,sample', [('kinetics', False), ('kinetics', True), ('pathway', True)])
def test_chemostat_workbook_supports_both_routes_and_sampling(tmp_path, variant, sample):
    """Execute the full workbook, including the exported pre/post sample rows.

    Parameters
    ----------
    tmp_path : Path
        Disposable output folder.
    variant : str
        Direct kinetics or reduced-pathway input selection.
    sample : bool
        Add the documented 10% midpoint sample.
    """
    notebook = EXAMPLES / 'generic_chemostat/workflow.ipynb'
    input_source = notebook.parent / ('inputs/kinetics' if variant == 'kinetics' else 'inputs')
    result = subprocess.run([sys.executable, '-c',
        "import runpy, pathlib, sys; "
        "runpy.run_path(sys.argv[1])['_check_notebook'](pathlib.Path(sys.argv[2]), "
        "pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), sys.argv[5] == 'True')",
        str(Path(__file__).resolve()), str(tmp_path), str(notebook), str(input_source), str(sample)],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
