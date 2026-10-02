"""Example inputs and notebook exports are reproducible without saved outputs."""

import json
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


def test_only_declared_examples_are_present():
    folders = sorted(path.name for path in EXAMPLES.iterdir() if path.is_dir() and not path.name.startswith("__"))
    assert folders == ["batch_ecoli_dfba", "ecoli_ye_fed_batch", "generic_batch", "generic_fed_batch"]


@pytest.mark.parametrize('name', ['batch_ecoli_dfba', 'ecoli_ye_fed_batch',
                                  'generic_batch', 'generic_fed_batch'])
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
    for filename in ('README.md', 'workflow.ipynb', 'inputs/case.json',
                     'inputs/mechanism.json', 'inputs/thermo.json'):
        assert (folder / filename).is_file()
    notebook = json.loads((folder / 'workflow.ipynb').read_text())
    namespace = {'__name__': '__notebook__'}
    try:
        for step, cell in enumerate((c for c in notebook['cells'] if c['cell_type'] == 'code'), 1):
            exec(compile(''.join(cell['source']), str(folder / 'workflow.ipynb'), 'exec'), namespace)
            if step == 1:
                namespace['EXPORT_DIR'] = tmp_path
        output = tmp_path / 'F10' if name == 'ecoli_ye_fed_batch' else tmp_path
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
