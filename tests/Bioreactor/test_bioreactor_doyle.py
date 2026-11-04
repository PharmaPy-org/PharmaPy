"""Textbook continuous-culture histories against independent balance integrations."""

import json
from pathlib import Path
import runpy

import pytest


EXAMPLE = Path(__file__).parents[2] / 'examples/bioreactors/Doyle_Exercise_4_15'


@pytest.mark.parametrize('scenario_index', range(3))
def test_doyle_independent_trajectories_and_refinement(scenario_index, tmp_path):
    """Check nominal and dilution-step histories using the declared error budgets.

    Parameters
    ----------
    scenario_index : int
        Index of the nominal, increased, or decreased dilution scenario.
    tmp_path : pathlib.Path
        Disposable destination for generated benchmark evidence.
    """
    specification = json.loads((EXAMPLE / 'benchmark.json').read_text())
    runner = runpy.run_path(str(EXAMPLE / 'run_benchmarks.py'))
    result = runner['run_scenario'](
        specification['scenarios'][scenario_index], specification, tmp_path)
    assert result['checks']
    assert result['passed'], result
