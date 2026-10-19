"""Example inputs and notebook exports are reproducible without saved outputs."""

import json
from .input_helpers import read_input_json
import matplotlib.pyplot as plt
import pytest
from pathlib import Path


ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples/bioreactors"


def test_bioreactor_package_exposes_only_forward_workflows():
    import PharmaPy.Bioreactors as package
    import importlib.util

    assert callable(package.build_bioreactor)
    for name in ('estimation', 'design', 'experiments'):
        assert importlib.util.find_spec('PharmaPy.Bioreactors.' + name) is None
    for name in ('estimate_bioreactor_parameters', 'evaluate_bioreactor_design',
                 'BioreactorEstimationProblem', 'BioreactorDesignProblem'):
        assert not hasattr(package, name)


def test_declared_examples_are_present():
    """Require distributed examples without depending on local scratch folders."""
    for name in ('Doyle_Exercise_4_15', 'batch_ecoli_dfba', 'ecoli_ye_fed_batch/reconciliation',
                 'generic_batch', 'generic_chemostat', 'generic_fed_batch'):
        assert (EXAMPLES / name / 'README.md').is_file()


@pytest.mark.parametrize('name', ['batch_ecoli_dfba', 'ecoli_ye_fed_batch/reconciliation',
                                  'generic_batch', 'generic_chemostat', 'generic_fed_batch',
                                  'Doyle_Exercise_4_15'])
def test_core_examples_have_forward_inputs_and_outputs(name, tmp_path):
    """Execute the exported client using disposable output storage.

    Parameters
    ----------
    name : str
        Retained example directory.
    tmp_path : pathlib.Path
        Isolated pytest export destination.
    """
    folder = EXAMPLES / name
    notebook_name = 'workbook.ipynb' if name == 'Doyle_Exercise_4_15' else 'workflow.ipynb'
    for filename in ('README.md', notebook_name,
                     'inputs/mechanism.json'):
        assert (folder / filename).is_file()
    thermo = folder.parent / 'inputs/thermo.json' if name.endswith('/reconciliation') else folder / 'inputs/thermo.json'
    assert thermo.is_file()
    notebook = read_input_json(folder / notebook_name)
    namespace = {'__name__': '__notebook__'}
    try:
        for step, cell in enumerate((c for c in notebook['cells'] if c['cell_type'] == 'code'), 1):
            exec(compile(''.join(cell['source']), str(folder / notebook_name), 'exec'), namespace)
            if step == 1:
                namespace['EXPORT_DIR'] = tmp_path
        output = tmp_path / 'F10' if name == 'ecoli_ye_fed_batch/reconciliation' else tmp_path
        for filename in ('trajectories.csv', 'trajectories.png'):
            assert (output / filename).stat().st_size > 0
    finally:
        plt.close('all')


def test_operational_modules_contain_no_flagship_identity_or_reaction_ids():
    forbidden = ("mahadevan", "reddy", "escherichia", "chinese hamster", "R047", "GLC")
    paths = [
        *list((ROOT / "PharmaPy/Bioreactors").glob("*.py")),
        *list((ROOT / "PharmaPy/Metabolic").rglob("*.py")),
    ]
    text = "\n".join(path.read_text().lower() for path in paths)
    for token in forbidden:
        assert token.lower() not in text
