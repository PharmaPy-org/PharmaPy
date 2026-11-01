"""All distributed examples expose named choices without replacing model physics.

Native trajectories and workbook exports are qualified by the existing example,
Doyle, continuous-culture and calibration tests. These checks guard the user
entry points and ensure every maintained notebook describes kinetic selection.
"""

import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[2] / 'examples/bioreactors'
DIRECT = (
    ('Doyle_Exercise_4_15/inputs/mechanism.json', 'mu'),
    ('Doyle_Exercise_4_15/inputs/hybrid.json', 'mu_mechanistic'),
    ('generic_chemostat/inputs/kinetics/mechanism.json', 'biomass_rate'),
    ('ecoli_ye_fed_batch/reconciliation/inputs/mechanism.json', 'q_s'),
    ('ecoli_ye_fed_batch/reconciliation/inputs/F11/mechanism.json', 'q_s'),
)
NOTEBOOKS = (
    'Doyle_Exercise_4_15/workbook.ipynb', 'Doyle_Exercise_4_15/calibration.ipynb',
    'generic_chemostat/workflow.ipynb', 'generic_batch/workflow.ipynb',
    'generic_fed_batch/workflow.ipynb', 'batch_ecoli_dfba/workflow.ipynb',
    'ecoli_ye_fed_batch/reconciliation/workflow.ipynb',
)


@pytest.mark.unit
@pytest.mark.parametrize('relative_path,identifier', DIRECT)
def test_example_growth_or_uptake_selects_monod_by_name(relative_path, identifier):
    """Require the familiar name at each example's documented rate selection.

    Parameters
    ----------
    relative_path : str
        Model JSON path relative to the examples directory.
    identifier : str
        Rule whose direct or composed expression contains the selected form.
    """
    model = json.loads((EXAMPLES/relative_path).read_text())
    rule = next(r for r in model['model']['kinetics']['rule_graph']
                if r['identifier'] == identifier)
    assert '"relationship": "monod"' in json.dumps(rule['expression'])


@pytest.mark.unit
@pytest.mark.parametrize('relative_path', NOTEBOOKS)
def test_each_notebook_exposes_selection_and_catalog(relative_path):
    """Guard the discoverable selection instructions and executable catalog.

    Parameters
    ----------
    relative_path : str
        Maintained workbook/notebook relative path.
    """
    notebook = json.loads((EXAMPLES/relative_path).read_text())
    sources = [''.join(cell['source']) for cell in notebook['cells']]
    assert any('Kinetics selection' in source for source in sources)
    assert any('relationship_catalog()' in source for cell, source in
               zip(notebook['cells'], sources) if cell['cell_type'] == 'code')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == []
            assert cell['execution_count'] is None
