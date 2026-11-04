"""Neutral corrections in shipped kinetic inputs preserve native trajectories.

Compare complete accepted species-mass histories with the same declarations
without corrections, using real native reactors and reconciliation collaborators.
Pathway constraints deliberately retain their separate uptake interface.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor

ROOT = Path(__file__).parents[2]
KINETIC_INPUTS = (
    'examples/bioreactors/Doyle_Exercise_4_15/inputs/mechanism.json',
    'examples/bioreactors/Doyle_Exercise_4_15/inputs/hybrid.json',
    'examples/bioreactors/generic_chemostat/inputs/kinetics/mechanism.json',
    'examples/bioreactors/ecoli_ye_fed_batch/reconciliation/inputs/mechanism.json',
    'examples/bioreactors/ecoli_ye_fed_batch/reconciliation/inputs/F11/mechanism.json',
    'doc/online_docs/bioreactors/templates/kinetics-selection/mechanism.json',
)


def remove_corrections(node: object) -> int:
    """Remove correction wrappers from a copied JSON tree.

    Parameters
    ----------
    node : object
        JSON node; physical inputs and their units are otherwise untouched.

    Returns
    -------
    int
        Number of removed wrappers, used to guard against vacuous comparisons.
    """
    count = 0
    if isinstance(node, dict):
        if 'correction' in node:
            del node['correction']
            count += 1
        for value in node.values():
            count += remove_corrections(value)
    elif isinstance(node, list):
        for value in node:
            count += remove_corrections(value)
    return count


@pytest.mark.integration
@pytest.mark.parametrize('relative_path', KINETIC_INPUTS)
def test_example_zero_correction_preserves_complete_native_history(relative_path):
    """Check neutral defaults, actual graph use, ordering and mass trajectories.

    Parameters
    ----------
    relative_path : str
        Shipped combined input file; its explicit units define the model basis.
    """
    path = ROOT / relative_path
    configured = json.loads(path.read_text())
    assert configured['parameters']['corrections']['theta'] == 0.
    original = deepcopy(configured)
    uncorrected = deepcopy(configured)
    assert remove_corrections(uncorrected) > 0
    if 'ecoli_ye_fed_batch' in relative_path:
        thermo = ROOT / 'examples/bioreactors/ecoli_ye_fed_batch/inputs/thermo.json'
    else:
        thermo = path.parent / 'thermo.json'
    actual_assembly = build_bioreactor(configured, thermo_path=thermo)
    base_assembly = build_bioreactor(uncorrected, thermo_path=thermo)
    actual = actual_assembly.solve(verbose=False)
    expected = base_assembly.solve(verbose=False)
    assert actual_assembly.events == base_assembly.events
    assert len(actual) == len(expected) > 0
    for observed, baseline in zip(actual, expected):
        np.testing.assert_array_equal(observed.time, baseline.time)
        assert observed.mass_j_liquid0.size > 0
        # Same equations and solver inputs: allow only roundoff, below solver tolerances.
        np.testing.assert_allclose(observed.mass_j_liquid0, baseline.mass_j_liquid0,
                                   rtol=1e-12, atol=1e-14)  # [-], [kg]
    assert configured == original
