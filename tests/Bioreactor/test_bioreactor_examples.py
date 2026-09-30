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
    assert folders == ["batch_ecoli_dfba", "ecoli_ye_fed_batch", "generic_batch", "generic_fed_batch",
                       "mammalian_surrogate_fed_batch"]


def test_core_examples_have_forward_inputs_and_outputs():
    for name in ('batch_ecoli_dfba', 'ecoli_ye_fed_batch', 'generic_batch', 'generic_fed_batch'):
        folder = EXAMPLES / name
        for filename in ('README.md', 'workflow.ipynb', 'inputs/case.json',
                         'inputs/mechanism.json', 'inputs/thermo.json'):
            assert (folder / filename).is_file()
        outputs = folder / 'outputs'
        if name == 'ecoli_ye_fed_batch':
            outputs = outputs / 'F10'
        for filename in ('trajectories.csv', 'trajectories.png'):
            assert (outputs / filename).is_file()





def test_operational_modules_contain_no_flagship_identity_or_reaction_ids():
    forbidden = ("mahadevan", "reddy", "escherichia", "chinese hamster", "R047", "GLC")
    paths = [
        *list((ROOT / "PharmaPy/Bioreactors").glob("*.py")),
        *list((ROOT / "PharmaPy/Metabolic").rglob("*.py")),
    ]
    text = "\n".join(path.read_text().lower() for path in paths)
    for token in forbidden:
        assert token.lower() not in text
