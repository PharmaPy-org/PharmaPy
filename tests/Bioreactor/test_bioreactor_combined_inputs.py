"""Combined input packaging preserves legacy models and rejects ambiguous inputs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor, BioreactorUnitConverter
from PharmaPy.Bioreactors.input_format import normalize_forward_inputs, split_forward_inputs

EXAMPLES = Path(__file__).parents[2]/'examples/bioreactors'
INPUTS = tuple(EXAMPLES / folder / 'mechanism.json' for folder in (
    'generic_batch/inputs', 'generic_fed_batch/inputs',
    'generic_chemostat/inputs', 'generic_chemostat/inputs/kinetics',
))


@pytest.mark.parametrize('path', INPUTS, ids=lambda path: str(path.relative_to(EXAMPLES)))
def test_combined_and_split_inputs_build_identical_native_objects(path):
    document = json.loads(path.read_text())
    original = deepcopy(document)
    case, definition = split_forward_inputs(document)
    assert normalize_forward_inputs(document) == normalize_forward_inputs(case, definition)
    # Variants intentionally share the nearest parent thermodynamic definition.
    thermo = next(candidate for parent in path.parents
                  for candidate in (parent/'thermo.json', parent/'inputs/thermo.json')
                  if candidate.exists())
    combined = build_bioreactor(document, thermo_path=thermo)
    separate = build_bioreactor(case, definition, thermo)
    assert type(combined.unit) is type(separate.unit)
    np.testing.assert_array_equal(combined.phase.mass_j, separate.phase.mass_j)
    np.testing.assert_array_equal(combined.time_grid, separate.time_grid)
    assert combined.events == separate.events
    assert document == original
    BioreactorUnitConverter.from_inputs(path)


@pytest.mark.parametrize('name', ['generic_batch', 'generic_fed_batch', 'generic_chemostat'])
def test_combined_histories_are_identical_to_separate_inputs(name):
    folder = EXAMPLES/name/'inputs'
    document = json.loads((folder/'mechanism.json').read_text())
    case, definition = split_forward_inputs(document)
    combined = build_bioreactor(document, thermo_path=folder/'thermo.json').solve()
    separate = build_bioreactor(case, definition, folder/'thermo.json').solve()
    assert len(combined) == len(separate)
    for actual, expected in zip(combined, separate):
        for field in ('time', 'mass_j_liquid0', 'vessel_vol'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize('field', ['operation', 'numerical'])
def test_missing_combined_sections_are_rejected(field):
    document = json.loads((EXAMPLES/'generic_batch/inputs/mechanism.json').read_text())
    del document[field]
    with pytest.raises(ValueError, match='combined input'):
        normalize_forward_inputs(document)


def test_combined_configuration_has_no_silent_precedence():
    document = json.loads((EXAMPLES/'generic_batch/inputs/mechanism.json').read_text())
    case, _ = split_forward_inputs(document)
    with pytest.raises(ValueError, match='alone'):
        normalize_forward_inputs(case, document)
    for key, value in [('numerics', {}), ('biology', {})]:
        invalid = dict(document, **{key: value})
        with pytest.raises(ValueError, match='unsupported'):
            normalize_forward_inputs(invalid)
    shuffled = dict(reversed(list(document.items())))
    assert normalize_forward_inputs(shuffled) == normalize_forward_inputs(document)
