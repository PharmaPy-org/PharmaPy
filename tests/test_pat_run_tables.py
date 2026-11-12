"""Layout-described table reading, the common run table and the caffeine dataset description.

The thermostat-log fixture is the first seven lines of a real log; expected values
are typed from those lines.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from examples.pat import table_layout
from examples.pat.run_table import RunTable

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parent.parent
CASE = REPO_ROOT / "examples" / "pat" / "caffeine"
LOG_EXCERPT = Path(__file__).parent / "fixtures" / "pat" / "huber_log_excerpt.csv"
ZERO_CELSIUS = 273.15  # [K]


def _log_layout() -> table_layout.TableLayout:
    return table_layout.load_layout(CASE / "huber_log.layout.json")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "log.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_thermostat_log_is_read_in_seconds_and_kelvin():
    values, units = table_layout.read_table(LOG_EXCERPT, _log_layout())

    assert units == {"time": "s", "jacket_temperature": "K", "external_temperature": "K", "setpoint_temperature": "K"}
    np.testing.assert_allclose(values["time"], [0.380, 2.380, 4.380, 6.380, 8.379])
    np.testing.assert_allclose(values["jacket_temperature"] - ZERO_CELSIUS, [19.840, 19.830, 19.830, 19.840, 19.840])
    np.testing.assert_allclose(values["external_temperature"] - ZERO_CELSIUS, [20.150] * 5)
    np.testing.assert_allclose(values["setpoint_temperature"] - ZERO_CELSIUS, [20.000] * 5)


def test_columns_are_found_by_title_whatever_their_order(tmp_path):
    reordered = _write(tmp_path, "note,H01:T Setpt,Time, H01:External T ,H01:Jacket T\n"
                                 "-,C,min,C,C\n"
                                 "a,30.0,0.5,25.5,21.0\n"
                                 "b,31.0,1.5,26.5,22.0\n")
    layout = _log_layout()
    minutes = table_layout.TableLayout(
        columns=tuple(table_layout.ColumnLayout(c.name, c.header, "min", c.role) if c.role == "time" else c for c in layout.columns),
        units_row=1, first_data_row=2)

    values, units = table_layout.read_table(reordered, minutes)

    np.testing.assert_allclose(values["time"], [30.0, 90.0])  # [s]
    np.testing.assert_allclose(values["jacket_temperature"], [294.15, 295.15])  # [K]
    np.testing.assert_allclose(values["external_temperature"], [298.65, 299.65])  # [K]
    np.testing.assert_allclose(values["setpoint_temperature"], [303.15, 304.15])  # [K]
    assert units["time"] == "s"


@pytest.mark.parametrize(
    "text, message",
    [
        ("Time,H01:Jacket T,H01:T Setpt\ns,C,C\n0,20,20\n", "exactly one column titled 'H01:External T', found 0"),
        ("Time,H01:Jacket T,H01:External T,H01:T Setpt\ns,C,F,C\n0,20,68,20\n", "is in 'F' but the layout declares 'C'"),
        ("Time,H01:Jacket T,H01:External T,H01:T Setpt\ns,C,C,C\n0,20,20,20\n0,20,20,20\n", "does not increase strictly"),
        ("Time,H01:Jacket T,H01:External T,H01:T Setpt\ns,C,C,C\n0,20,293.15,20\n", "outside the declared range 0 to 100"),
        ("Time,H01:Jacket T,H01:External T,H01:T Setpt\ns,C,C,C\n0,20,nan,20\n", "non-finite value"),
        ("Time,H01:Jacket T,H01:External T,H01:T Setpt\ns,C,C,C\n0,20,,20\n", "missing or non-numeric cell"),
        ("Time,H01:Jacket T,H01:External T,H01:T Setpt\ns,C,C,C\n", "has no data rows"),
    ],
)
def test_a_file_that_does_not_match_the_layout_is_refused(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        table_layout.read_table(_write(tmp_path, text), _log_layout())


@pytest.mark.parametrize(
    "columns, message",
    [
        ({"t": {"header": "Time", "unit": "s", "role": "time"}, "x": {"header": "X", "unit": "F"}}, "unit 'F' is not one of"),
        ({"t": {"header": "Time", "unit": "s"}}, "exactly one column with role 'time'"),
        ({"t": {"header": "Time", "unit": "s", "role": "clock"}}, "role 'clock' is not one of"),
        ({"t": {"header": "Time", "unit": "s", "role": "time", "scale": 2}}, "unknown keys \\['scale'\\]"),
    ],
)
def test_an_invalid_layout_description_is_refused(tmp_path, columns, message):
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"columns": columns}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        table_layout.load_layout(path)


def test_run_table_shifts_each_series_onto_the_run_clock():
    table = RunTable("example")

    table.add("temperature", "K", [0.0, 2.0, 4.0], [293.0, 294.0, 295.0])
    counts = table.add("counts", "counts", [10.0, 20.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], clock_offset=-170.0)

    assert list(table.series) == ["temperature", "counts"]
    np.testing.assert_allclose(counts.time, [-160.0, -150.0])
    assert counts.values.shape == (2, 3) and counts.unit == "counts"
    np.testing.assert_allclose(table.series["temperature"].time, [0.0, 2.0, 4.0])


@pytest.mark.parametrize(
    "time, values, message",
    [
        ([0.0, 1.0], [1.0, 2.0, 3.0], "first axes must match"),
        ([0.0, 0.0], [1.0, 2.0], "do not increase strictly"),
        ([0.0, 1.0], [1.0, np.inf], "non-finite"),
    ],
)
def test_run_table_refuses_inconsistent_series(time, values, message):
    with pytest.raises(ValueError, match=message):
        RunTable("example").add("x", "-", time, values)


def test_run_table_refuses_a_duplicate_name():
    table = RunTable("example")
    table.add("x", "-", [0.0], [1.0])

    with pytest.raises(ValueError, match="already has a series named 'x'"):
        table.add("x", "-", [0.0], [1.0])


def test_caffeine_dataset_description_is_self_consistent():
    description = json.loads((CASE / "runs.json").read_text(encoding="utf-8"))
    runs = description["runs"]
    yield_g = description["charge"]["theoretical_yield_g"]  # [g]

    assert (CASE / description["temperature_log"]["layout"]).is_file()
    assert len({run["id"] for run in runs}) == len({run["folder"] for run in runs}) == 4
    # Recipe sheet: 16.5434 g charged, 5.1169 g expected to stay dissolved at 20 C.
    assert yield_g == pytest.approx(16.5434 - 5.1169, abs=1e-4)
    for run in runs:
        assert run["seed_mass_g"] == pytest.approx(run["seed_percent_of_yield"] / 100.0 * yield_g, abs=5e-5)
    assert sorted((run["seed_percent_of_yield"], run["final_ramp_rate_C_per_min"]) for run in runs) == [
        (1.0, 0.1), (1.0, 0.5), (5.0, 0.1), (5.0, 0.5)]
