"""An independent, forward-only example with an analytical mass-balance check."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("name, uptake, feed_volume", [
    ("generic_batch", 2.0, None),
    ("generic_batch", 1.0, None),
    ("generic_fed_batch", 2.0, .1),
    ("generic_fed_batch", 1.0, .1),
    ("generic_fed_batch", 2.0, .05),
])
def test_generic_notebook_and_editable_inputs(monkeypatch, tmp_path, name, uptake, feed_volume):
    example = ROOT / "examples/bioreactors" / name
    document = json.loads((example / "workflow.ipynb").read_text())
    namespace = {"__name__": "__notebook__"}
    monkeypatch.chdir(ROOT)
    cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 6
    try:
        for step, cell in enumerate(cells, 1):
            source = "".join(cell["source"])
            exec(compile(source, f"generic-workflow:step-{step}", "exec"), namespace)
            if step == 1:
                namespace["EXPORT_DIR"] = tmp_path
            elif step == 2:
                namespace["config"]["parameters"]["rates"]["uptake"]["nutrient"] = uptake
                if feed_volume is not None:
                    namespace["config"]["recipes"]["fed_batch"][0]["volume_l"] = feed_volume
        rows = namespace['rows']
        times = np.array([row['time_h'] for row in rows])
        biomass = .1 * np.exp(uptake / 10. * times)
        added = np.zeros(len(rows)) if feed_volume is None else np.array([row['feed_applied'] for row in rows])
        feed = 0. if feed_volume is None else feed_volume
        nutrient = 10. + added * 50. * feed - 10. * (biomass - .1)
        total_mass = 1.0001 + added * 1.005 * feed
        np.testing.assert_allclose([r['biomass_gdw'] for r in rows], biomass, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose([r['nutrient_mmol'] for r in rows], nutrient, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose([r['liquid_plus_biomass_kg'] for r in rows], total_mass, atol=1e-10, rtol=0)
        np.testing.assert_allclose([r['liquid_volume_l'] for r in rows], total_mass - biomass / 1000., rtol=1e-6)
        assert len(rows) == (61 if feed_volume is None else 62)
        assert (tmp_path / 'trajectories.csv').is_file()
    finally:
        plt.close(namespace['figure'])
