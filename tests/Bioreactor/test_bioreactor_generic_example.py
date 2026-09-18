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
    assert len(cells) == 7
    for step, cell in enumerate(cells, 1):
        source = "".join(cell["source"])
        assert source.startswith(f"# STEP {step}:")
        exec(compile(source, f"generic-workflow:step-{step}", "exec"), namespace)
        if step == 1:
            assert namespace["WRITE_ARTIFACTS"] is False
            namespace["EXPORT_DIR"] = tmp_path
        elif step == 2:
            namespace["config"]["model"]["parameters"]["nutrient_uptake_max"] = uptake
            if feed_volume is not None:
                namespace["config"]["recipes"]["fed_batch"][0]["volume_l"] = feed_volume
    try:
        report = namespace["summary"]
        assert report["status"] == "PASS" and all(report["checks"].values())
        assert report["native_unit"] == ("BatchReactor" if feed_volume is None else "SemiBatchReactor")
        assert report["recorded_rows"] == (61 if feed_volume is None else 62)
        assert report["final"]["biomass_gdw"] == pytest.approx(
            .1 * np.exp(uptake / 10.0 * 6.0), rel=1e-6)
        assert not list(tmp_path.iterdir())
        if feed_volume is not None:
            before, after = report["feed"]["pre"], report["feed"]["post"]
            assert after["nutrient_mmol"] - before["nutrient_mmol"] == pytest.approx(50.0 * feed_volume)
            assert after["liquid_volume_l"] - before["liquid_volume_l"] == pytest.approx(1.005 * feed_volume)
            assert after["biomass_gdw"] == pytest.approx(before["biomass_gdw"], abs=1e-10)
            assert after["biomass_gdw_l"] < before["biomass_gdw_l"]
        if uptake == 2.0 and feed_volume in (None, .1):
            stored = np.genfromtxt(example / "outputs/trajectories.csv",
                                   delimiter=",", names=True)
            for name in stored.dtype.names:
                np.testing.assert_allclose([row[name] for row in namespace["rows"]],
                                           stored[name], rtol=1e-6, atol=1e-10)
            reference = json.loads((example / "outputs/summary.json").read_text())
            assert reference["effective_inputs"] == report["effective_inputs"]
            assert reference["thermo_sha256"] == report["thermo_sha256"]
            assert (example / "outputs/trajectories.png").stat().st_size > 0
    finally:
        plt.close(namespace["figure"])
