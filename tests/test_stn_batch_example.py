"""Exercise the documented STN export with two real CVode batch solves."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip("assimulo")


def test_solved_batch_example_exports_reference_stn(tmp_path):
    """Check fixed batch masses, material arcs and rounding in the CLI output."""
    checkout = Path(__file__).resolve().parents[1]
    output = tmp_path / "batch.json"
    subprocess.run(
        [sys.executable, "-m", "doc.online_docs.examples.stn_batch_example",
         str(output)],
        cwd=checkout, check=True, capture_output=True, text=True,
    )
    stn = json.loads(output.read_text(encoding="utf-8"))

    assert stn["I"] == ["R01", "R02"]
    assert stn["J"] == ["reactor-1", "reactor-2"]
    assert stn["K"] == ["feed", "intermediate", "product"]
    assert stn["delta"] == pytest.approx(20.0)  # [s], example grid width
    # 25 s needs two 20 s periods and 43 s needs three, so both fit in five.
    assert stn["lastT"] == 5  # [-]
    task_units = [["R01", "reactor-1"], ["R02", "reactor-2"]]
    assert [entry["index"] for entry in stn["tau_p"]] == task_units
    assert [entry["value"] for entry in stn["tau_p"]] == pytest.approx(
        [25.0, 43.0])  # [s], stage run durations requested by the example
    assert stn["tau"] == [
        {"index": task_units[0], "value": 2},
        {"index": task_units[1], "value": 3},
    ]  # [-], minimal durations on the 20 s grid
    assert stn["I_i_j_prod"] == [
        {"index": index, "value": 1} for index in task_units]

    inputs = [["R01", "feed"], ["R02", "intermediate"]]
    outputs = [["R01", "intermediate"], ["R02", "product"]]
    for direction, indices in (("minus", inputs), ("plus", outputs)):
        assert stn[f"I_i_k_{direction}"] == [
            {"index": index, "value": 1} for index in indices]
        ratios = stn[f"rho_{direction}"]
        assert [entry["index"] for entry in ratios] == indices
        assert [entry["value"] for entry in ratios] == pytest.approx([1.0, 1.0])
        # [-], one entire liquid batch on each side of the mass-balanced task.

    # [kg], independently from 2 L * (0.1*105.52 + 0.2*99.1741 + 5*72.107) g/L
    # with 1000 g/kg. The shipped molar masses satisfy MW(C) = MW(A) + MW(B).
    batch_kg = 0.78184364
    for capacity in ("beta_min", "beta_max"):
        assert [entry["index"] for entry in stn[capacity]] == task_units
        assert [entry["value"] for entry in stn[capacity]] == pytest.approx(
            [batch_kg, batch_kg])
    assert [entry["index"] for entry in stn["gamma"]] == stn["K"]
    assert [entry["value"] for entry in stn["gamma"]] == pytest.approx(
        [batch_kg, batch_kg, batch_kg])
    assert stn["S0"] == [{"index": "feed", "value": pytest.approx(batch_kg)}]
    assert stn["demand"] == [
        {"index": ["product", 5], "value": pytest.approx(batch_kg)}]
    assert stn["cost"] == []
