"""Read example configurations as legacy test inputs without changing fixtures."""
import json
from pathlib import Path

from PharmaPy.Bioreactors.input_format import split_forward_inputs


def read_input_json(path: Path):
    """Read JSON, unpacking migrated examples for existing schema-2 tests.

    Parameters
    ----------
    path : Path
        Input path. An absent case.json is read from its combined mechanism.

    Returns
    -------
    object
        Original JSON content, or the corresponding schema-2 document with
        all physical and numerical values unchanged.
    """
    path = Path(path)
    if path.name == 'case.json' and not path.exists():
        return split_forward_inputs(json.loads(path.with_name('mechanism.json').read_text()))[0]
    document = json.loads(path.read_text())
    if path.name == 'mechanism.json' and document.get('schema_version') == 3:
        return split_forward_inputs(document)[1]
    return document
