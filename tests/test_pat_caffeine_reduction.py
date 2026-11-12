"""Reduction of the caffeine runs: its steps on fixtures, and the committed reduced tables.

The raw runs are not in the repository. The steps are exercised on the excerpts in
``tests/fixtures/pat``; the committed tables are checked against the run design in
``runs.json`` and against events documented in ``examples/pat/caffeine/README.md``.
"""
import copy
import datetime
from pathlib import Path

import numpy as np
import pytest

from examples.pat import block_package, ic_fbrm, ic_spectra, table_layout
from examples.pat.caffeine import reduce_runs
from examples.pat.run_table import block_mean

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "pat"
CASE = reduce_runs.CASE
DESCRIPTION = reduce_runs.load_description()
RUNS = {run["id"]: run for run in DESCRIPTION["runs"]}
ZERO_CELSIUS = 273.15  # [K]
SEEDING_NODE = 8460.0  # [s] node of the setpoint program inside the 37 C hold
FINAL_RAMP_START = 9360.0  # [s] end of the 37 C hold in the setpoint program


def _reduced(run_id: str, kind: str) -> dict:
    layout = table_layout.load_layout(CASE / ("reduced_%s.layout.json" % kind))
    values, _ = table_layout.read_table(CASE / "reduced" / ("%s_%s.csv" % (run_id, kind)), layout)
    return values


def test_fbrm_table_puts_samples_on_the_run_clock_with_vendor_statistics():
    times, raw = ic_fbrm.read_samples(FIXTURES / "fbrm_three_samples.icFbrm")
    run = RUNS["high_seed_slow_cool"]  # temperature log started 170 s after the FBRM

    origin = reduce_runs.run_clock_origin(times[0], run, DESCRIPTION)
    columns, units = reduce_runs.fbrm_table(times, raw, origin, DESCRIPTION)

    assert list(columns) == ["time", "counts_1_to_10um", "counts_10_to_100um", "counts_100_to_1000um", "mean_chord_square_weighted"]
    assert units == {"time": "s", "counts_1_to_10um": "counts", "counts_10_to_100um": "counts",
                     "counts_100_to_1000um": "counts", "mean_chord_square_weighted": "um"}
    # The first sample closes a 10 s interval that began 170 s before the log: 10 - 170 s.
    # The others follow 4 h 27 min 1.612152 s and 10 h 0 min 49.715862 s later.
    np.testing.assert_allclose(columns["time"], [-160.0, -160.0 + 16021.612152, -160.0 + 36049.715862], atol=1e-6)
    # Statistics stored by the vendor software for these samples (see test_pat_signal_statistics).
    np.testing.assert_allclose(columns["counts_10_to_100um"], [25075.2, 33528.3, 28549.4], rtol=1e-2)
    np.testing.assert_allclose(columns["mean_chord_square_weighted"], [121.868, 77.0612, 90.7105], rtol=5e-4)


def test_ir_table_sorts_by_time_and_reports_band_height_and_solvent_absorbance():
    infos = ic_spectra.spectrum_infos((FIXTURES / "spectrum_info_tags.xml").read_text(encoding="utf-8"))
    stream = block_package.read_package((FIXTURES / "ir_head_blocks.iCIR").read_bytes())["/ReactionSpectras.vector"][1]
    records = dict(ic_spectra.spectrum_records(stream))
    with_metadata = [identifier for identifier in records if identifier in infos]
    shuffled = {identifier: records[identifier] for identifier in reversed(with_metadata)}
    first_fbrm_sample = datetime.datetime(2026, 10, 24, 0, 46, 37)  # local time, as in the FBRM export
    origin = reduce_runs.run_clock_origin(first_fbrm_sample, RUNS["high_seed_slow_cool"], DESCRIPTION)

    columns, units = reduce_runs.ir_table(infos, shuffled, origin, utc_offset_h=-4.0, description=DESCRIPTION)

    # Run clock zero is 00:46:37 - 10 s + 170 s = 00:49:17 local. The spectra were taken
    # at 04:29:10.257475, 04:30:11.952319 and 04:31:10.578463 UTC, four hours ahead.
    np.testing.assert_allclose(columns["time"], [-1206.742525, -1145.047681, -1086.421537], atol=1e-6)
    # Export absorbances at 1760, 1720, 1700 and 1640 cm-1 of the same spectra.
    a1760 = np.array([0.146599612457919, 0.147050707666293, 0.148026113384757])
    a1720 = np.array([0.248405187363523, 0.248816188191783, 0.250148186896707])
    a1700 = np.array([0.364370714213183, 0.365671123431154, 0.367121402604299])
    np.testing.assert_allclose(columns["band_height"], a1700 - 1.5 * a1720 + 0.5 * a1760, atol=1e-12)
    export = np.loadtxt(FIXTURES / "ir_export_columns.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(columns["solvent_absorbance"], export[export[:, 0] == 1640.0, 1:].ravel(), atol=1e-12)
    assert units == {"time": "s", "band_height": "-", "solvent_absorbance": "-"}


def test_ir_table_refuses_a_spectrum_without_metadata():
    infos = ic_spectra.spectrum_infos((FIXTURES / "spectrum_info_tags.xml").read_text(encoding="utf-8"))
    stream = block_package.read_package((FIXTURES / "ir_head_blocks.iCIR").read_bytes())["/ReactionSpectras.vector"][1]

    with pytest.raises(ValueError, match="has no metadata entry"):
        reduce_runs.ir_table(infos, dict(ic_spectra.spectrum_records(stream)), datetime.datetime(2026, 10, 24), -4.0, DESCRIPTION)


def test_sample_time_without_fraction_is_parsed():
    assert reduce_runs._parse_utc("2026-10-24T04:29:10Z") == datetime.datetime(2026, 10, 24, 4, 29, 10)
    assert reduce_runs._parse_utc("2026-10-24T04:29:10.2574757Z") == datetime.datetime(2026, 10, 24, 4, 29, 10, 257475)


def test_temperature_table_averages_blocks_of_the_log():
    description = copy.deepcopy(DESCRIPTION)
    description["temperature_log"]["block_samples"] = 3  # the excerpt has five rows

    columns, units = reduce_runs.temperature_table(FIXTURES / "huber_log_excerpt.csv", description)

    # Rows 1 to 3 of the excerpt: 0.380, 2.380, 4.380 s and jacket 19.840, 19.830, 19.830 C.
    np.testing.assert_allclose(columns["time"], [2.380])
    np.testing.assert_allclose(columns["jacket_temperature"], [(19.840 + 19.830 + 19.830) / 3.0 + ZERO_CELSIUS])
    np.testing.assert_allclose(columns["external_temperature"], [20.150 + ZERO_CELSIUS])
    assert units["external_temperature"] == "K"


def test_block_mean_drops_the_incomplete_last_block():
    values = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0], [100.0, 1000.0]])

    np.testing.assert_allclose(block_mean(values, 2), [[2.0, 20.0], [6.0, 60.0]])
    np.testing.assert_allclose(block_mean(values[:, 0], 5), [23.2])
    with pytest.raises(ValueError, match="does not fit"):
        block_mean(values, 6)


def test_missing_values_are_written_empty_and_read_as_nan_only_where_allowed(tmp_path):
    path = tmp_path / "table.csv"
    columns = {"time": np.array([0.0, 10.0]), "chord": np.array([np.nan, 52.5])}
    table_layout.write_table(path, columns, {"time": "s", "chord": "um"}, {"time": 1, "chord": 1}, missing=("chord",))
    time = table_layout.ColumnLayout("time", "time", "s", "time")

    allowed = table_layout.TableLayout((time, table_layout.ColumnLayout("chord", "chord", "um", minimum=1.0, allow_missing=True)), units_row=1, first_data_row=2)
    values, _ = table_layout.read_table(path, allowed)

    assert path.read_text(encoding="utf-8").splitlines()[2] == "0.0,"
    assert np.isnan(values["chord"][0]) and values["chord"][1] == 52.5
    strict = table_layout.TableLayout((time, table_layout.ColumnLayout("chord", "chord", "um")), units_row=1, first_data_row=2)
    with pytest.raises(ValueError, match="missing or non-numeric cell"):
        table_layout.read_table(path, strict)
    with pytest.raises(ValueError, match="not declared as missing"):
        table_layout.write_table(path, columns, {"time": "s", "chord": "um"}, {"time": 1, "chord": 1})


def test_a_time_column_cannot_allow_missing_values(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text('{"columns": {"t": {"header": "t", "unit": "s", "role": "time", "allow_missing": true}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="time column cannot allow missing"):
        table_layout.load_layout(path)


@pytest.mark.parametrize("run_id", sorted(RUNS))
def test_reduced_tables_follow_the_run_design(run_id):
    run = RUNS[run_id]
    temperature = _reduced(run_id, "temperature")
    fbrm = _reduced(run_id, "fbrm")
    total = fbrm["counts_1_to_10um"] + fbrm["counts_10_to_100um"] + fbrm["counts_100_to_1000um"]  # [counts]
    block_seconds = DESCRIPTION["temperature_log"]["block_samples"] * 2.0  # [s] the log has a row every 2 s

    assert np.median(np.diff(temperature["time"])) == pytest.approx(block_seconds, abs=0.1)
    assert np.median(np.diff(fbrm["time"])) == pytest.approx(DESCRIPTION["fbrm"]["sample_interval_s"], abs=0.5)
    # Program: 45 C at the top, 37 C at seeding, then a ramp at the run's rate.
    assert temperature["external_temperature"].max() == pytest.approx(45.0 + ZERO_CELSIUS, abs=0.1)
    in_hold = (temperature["time"] > 7800.0) & (temperature["time"] < FINAL_RAMP_START)
    assert np.median(temperature["setpoint_temperature"][in_hold]) == pytest.approx(DESCRIPTION["seeding"]["temperature_C"] + ZERO_CELSIUS, abs=0.01)
    on_ramp = (temperature["time"] > FINAL_RAMP_START + 300.0) & (temperature["time"] < FINAL_RAMP_START + 1500.0)
    slope = np.polyfit(temperature["time"][on_ramp], temperature["setpoint_temperature"][on_ramp], 1)[0]  # [K/s]
    assert -slope * 60.0 == pytest.approx(run["final_ramp_rate_C_per_min"], rel=0.02)
    # Clear solution between the 45 C hold and seeding, then the seeding spike 2.5 to
    # 5.5 min after the program node, then a product slurry (documented in the case README).
    clear = (fbrm["time"] > 6000.0) & (fbrm["time"] < 8400.0)
    assert np.median(total[clear]) < 50.0
    after_node = (fbrm["time"] > SEEDING_NODE) & (fbrm["time"] < FINAL_RAMP_START)
    spike_time = fbrm["time"][after_node][np.argmax(total[after_node])]
    assert total[after_node].max() > 300.0
    assert 150.0 < spike_time - SEEDING_NODE < 330.0
    assert np.median(total[fbrm["time"] > fbrm["time"][-1] - 600.0]) > 20000.0
    # A higher seed load gives a larger spike.
    assert (total[after_node].max() > 700.0) == (run["seed_percent_of_yield"] == 5.0)


def test_reduced_ir_shows_dissolution_where_the_probe_was_immersed():
    plateau = {}
    for run_id in RUNS:
        ir = _reduced(run_id, "ir")
        program = ir["time"] > 0.0
        slurry = (ir["time"] > 600.0) & (ir["time"] < 1800.0)  # 20 C, solid present
        dissolved = (ir["time"] > 5000.0) & (ir["time"] < 6600.0)  # 45 C hold
        plateau[run_id] = (np.median(ir["band_height"][slurry]), np.median(ir["band_height"][dissolved]),
                           ir["solvent_absorbance"][program].min())

    for run_id in ("high_seed_slow_cool", "low_seed_slow_cool", "low_seed_fast_cool"):
        slurry_height, dissolved_height, solvent_minimum = plateau[run_id]
        assert solvent_minimum > 0.85
        assert 0.085 < slurry_height < 0.100 and 0.128 < dissolved_height < 0.138
    # Run 4: the probe was not in contact with the liquid, so the band barely responds.
    slurry_height, dissolved_height, solvent_minimum = plateau["high_seed_fast_cool"]
    assert solvent_minimum < 0.1 and dissolved_height < 0.02
