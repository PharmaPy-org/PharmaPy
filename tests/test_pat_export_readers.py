"""Vendor CSV export readers of ``PharmaPy.PATReaders`` and the reduction of a run from exports.

Fixtures in ``tests/fixtures/pat`` are excerpts of run 2026-10-23: the exports and
the native files hold the same three spectra and the same three FBRM samples.
Expected values are read off the export text or come from the native files.
"""
import copy
import datetime
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from examples.pat import dataset, reduce_runs, table_layout
from PharmaPy import PATReaders

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "pat"
CASE = Path(dataset.__file__).parent / "caffeine"
DESCRIPTION = dataset.load_description(CASE)
RUN = next(run for run in DESCRIPTION["runs"] if run["id"] == "high_seed_slow_cool")
IR_EXPORT, FBRM_EXPORT = FIXTURES / "ir_export_columns.csv", FIXTURES / "fbrm_export_rows.csv"
# Start of the iC IR experiment of the run. The sample time of every one of its 663
# spectra minus its export title lies between 04:28:18.67 and 04:28:19.67 UTC.
IR_START_UTC = datetime.datetime(2026, 10, 24, 4, 28, 19)
IR_START_LOCAL = "2026-10-24T00:28:19"  # the same instant in the lab's local time, four hours behind UTC
ONE_SECOND = np.timedelta64(1, "s")  # resolution of the export's time stamps
# The export prints 15 significant digits of absorbances below 1; see test_pat_native_readers.
ABSORBANCE_TOLERANCE = 1e-12  # [-]
# Largest relative difference between rebinned raw counts and the export over the
# 4313 samples of the run was 3.3e-3; see test_pat_native_readers.
CHANNEL_TOLERANCE = 5e-3  # [-]


def _native_spectra() -> PATReaders.ICSpectra:
    infos = PATReaders.spectrum_infos((FIXTURES / "spectrum_info_tags.xml").read_text(encoding="utf-8"))
    stream = PATReaders.read_package((FIXTURES / "ir_head_blocks.iCIR").read_bytes())["/ReactionSpectras.vector"][1]
    return PATReaders.spectra_from_records(infos, dict(PATReaders.spectrum_records(stream)))


def _edited(folder: Path, source: Path, old: str, new: str) -> Path:
    """Write a copy of an export with one piece of its text replaced."""
    text = source.read_text(encoding="utf-8")
    assert text.count(old) >= 1
    path = folder / source.name
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def _samples(edges: list, counts: list) -> PATReaders.FBRMSamples:
    return PATReaders.FBRMSamples(time=np.array([datetime.datetime(2026, 10, 24, 1, 0, 0)], dtype="datetime64[us]"),
                                  channel_edges=np.array(edges, dtype=float), counts=np.array([counts], dtype=float))


def test_ir_export_gives_the_spectra_of_the_native_file():
    native = _native_spectra()

    ir = PATReaders.read_ic_ir_export(IR_EXPORT, IR_START_UTC)

    assert ir.names == native.names == ("00:00:51", "00:01:53", "00:02:51")
    # The start plus the three titles.
    assert ir.time.dtype == np.dtype("datetime64[us]")
    assert ir.time.tolist() == [datetime.datetime(2026, 10, 24, 4, 29, 10), datetime.datetime(2026, 10, 24, 4, 30, 12),
                                datetime.datetime(2026, 10, 24, 4, 31, 10)]
    assert np.all(np.abs(native.time - ir.time) < ONE_SECOND)
    assert ir.wavenumber.shape == (589,) and ir.wavenumber[[0, 1, -1]].tolist() == [3000.0, 2996.0, 648.0]
    np.testing.assert_array_equal(ir.wavenumber, native.wavenumber)
    # First data row of the export: the absorbance at 3000 cm-1 of the three spectra.
    assert ir.absorbance.shape == (3, 589)
    assert ir.absorbance[:, 0].tolist() == [0.377040377806009, 0.382678510590497, 0.380032270470555]
    np.testing.assert_allclose(ir.absorbance, native.absorbance, rtol=0.0, atol=ABSORBANCE_TOLERANCE)
    assert (ir.value_units, ir.without_time, ir.software, ir.build) == ("Absorbance", (), "", "")


def test_ir_export_start_with_a_time_zone_is_converted_to_utc():
    eastern_daylight = datetime.timezone(datetime.timedelta(hours=-4))
    local_start = datetime.datetime(2026, 10, 24, 0, 28, 19, tzinfo=eastern_daylight)

    ir = PATReaders.read_ic_ir_export(IR_EXPORT, local_start)

    assert ir.time[0].item() == datetime.datetime(2026, 10, 24, 4, 29, 10)


def test_ir_export_reads_hours_that_keep_counting_past_a_day(tmp_path):
    ir = PATReaders.read_ic_ir_export(_edited(tmp_path, IR_EXPORT, "00:02:51", "25:00:10"), IR_START_UTC)

    assert ir.time[2].item() == datetime.datetime(2026, 10, 25, 5, 28, 29)


def test_exports_are_read_with_a_byte_order_mark_and_blank_lines(tmp_path):
    ir_path, fbrm_path = tmp_path / "ir.csv", tmp_path / "fbrm.csv"
    ir_path.write_bytes(b"\xef\xbb\xbf" + IR_EXPORT.read_bytes().replace(b"\n2996,", b"\n\n2996,", 1) + b"\n\n")
    fbrm_path.write_bytes(b"\xef\xbb\xbf" + FBRM_EXPORT.read_bytes().replace(b"\n2001,", b"\n\n2001,", 1) + b"\n")

    ir = PATReaders.read_ic_ir_export(ir_path, IR_START_UTC)
    fbrm = PATReaders.read_ic_fbrm_export(fbrm_path)

    np.testing.assert_array_equal(ir.absorbance, PATReaders.read_ic_ir_export(IR_EXPORT, IR_START_UTC).absorbance)
    np.testing.assert_array_equal(fbrm.counts, PATReaders.read_ic_fbrm_export(FBRM_EXPORT).counts)
    assert ir.absorbance.shape == (3, 589) and fbrm.counts.shape == (3, 100)


@pytest.mark.parametrize("old, new, message", [
    ("Wavenumbercm-1", "Raman Shift (cm-1)", "is not an iC IR export"),
    ("00:01:53", "10/24/2026 0:47", "Column title '10/24/2026 0:47' .* is not a time since the start"),
    ("00:01:53", "1.00:01:53", "is not a time since the start"),
    ("00:01:53,00:02:51", "00:02:51,00:01:53", "do not increase"),
    ("\n2996,0.368890587046429,", "\n2996,", "Data row 2 .* has 3 cells; the title row has 4"),
    ("0.368890587046429", "", "not a number"),
    ("0.368890587046429", "saturated", "not a number"),
    ("0.368890587046429", "nan", "not finite"),
])
def test_ir_export_refuses_a_file_it_cannot_place_in_time_or_read(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        PATReaders.read_ic_ir_export(_edited(tmp_path, IR_EXPORT, old, new), IR_START_UTC)


def test_ir_export_refuses_a_file_without_spectra_and_a_start_that_is_not_a_time(tmp_path):
    titles_only = tmp_path / "titles.csv"
    titles_only.write_text("Wavenumbercm-1,00:00:51\n", encoding="utf-8")
    axis_only = tmp_path / "axis.csv"
    axis_only.write_text("Wavenumbercm-1\n3000\n", encoding="utf-8")

    for path in (titles_only, axis_only):
        with pytest.raises(ValueError, match="holds no spectrum"):
            PATReaders.read_ic_ir_export(path, IR_START_UTC)
    with pytest.raises(TypeError, match="start_utc must be a datetime.datetime, not str"):
        PATReaders.read_ic_ir_export(IR_EXPORT, "2026-10-24T04:28:19")


def test_fbrm_export_gives_the_samples_of_the_native_file():
    native = PATReaders.read_ic_fbrm(FIXTURES / "fbrm_three_samples.icFbrm")

    fbrm = PATReaders.read_ic_fbrm_export(FBRM_EXPORT)

    # Date and time cells of samples 401, 2001 and 4001.
    assert fbrm.time.dtype == np.dtype("datetime64[us]")
    assert fbrm.time.tolist() == [datetime.datetime(2026, 10, 24, 1, 53, 23), datetime.datetime(2026, 10, 24, 6, 20, 24),
                                  datetime.datetime(2026, 10, 24, 11, 54, 13)]
    assert [time.replace(microsecond=0) for time in native.time.tolist()] == fbrm.time.tolist()
    # Row Ch.Bd: 101 boundaries of 100 channels.
    assert fbrm.channel_edges.shape == (101,) and fbrm.channel_edges[[0, 1, -2, -1]].tolist() == [1.0, 1.0715, 933.25, 1000.0]
    # First counts of each sample row, and the row totals.
    assert fbrm.counts.shape == (3, 100)
    assert fbrm.counts[:, 0].tolist() == [1.0258, 3.6459, 1.8881] and fbrm.counts[0, 1] == 1.0992
    assert fbrm.counts.sum(axis=1).tolist() == pytest.approx([36776.6, 51983.5, 40899.8], abs=0.1)
    rebinned = native.counts @ PATReaders.rebin_matrix(native.channel_edges, fbrm.channel_edges)
    assert (np.abs(rebinned - fbrm.counts) / np.maximum(fbrm.counts, 1.0)).max() < CHANNEL_TOLERANCE
    assert (fbrm.software, fbrm.build) == ("", "")


def test_fbrm_export_reads_an_afternoon_time(tmp_path):
    fbrm = PATReaders.read_ic_fbrm_export(_edited(tmp_path, FBRM_EXPORT, "6:20:24 AM", "6:20:24 PM"))

    assert fbrm.time[1].item() == datetime.datetime(2026, 10, 24, 18, 20, 24)


@pytest.mark.parametrize("old, new, message", [
    ("No Weight", "Square Weight", "holds counts with the weighting 'Square Weight'"),
    ("Ch.Bd", "Ch.Edges", "is not an iC FBRM export"),
    ("Ch.Bd,,,,1,1.0715,", "Ch.Bd,,,,1.0715,1,", "must be at least two increasing values"),
    ("401,10/24/2026,1:53:23 AM,,1.0258,", "401,10/24/2026,1:53:23 AM,,", "Sample 401 .* has 99 counts; the file states 100 channels"),
    ("10/24/2026,1:53:23 AM", "24/10/2026,1:53:23", "Sample 401 .* '24/10/2026 1:53:23', not written like"),
    ("401,10/24/2026,1:53:23 AM,,1.0258,", "401,10/24/2026,1:53:23 AM,,many,", "not a number"),
])
def test_fbrm_export_refuses_a_file_it_cannot_read_as_unweighted_counts(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        PATReaders.read_ic_fbrm_export(_edited(tmp_path, FBRM_EXPORT, old, new))


def test_fbrm_export_refuses_a_file_without_samples(tmp_path):
    header = "".join(FBRM_EXPORT.read_text(encoding="utf-8").splitlines(keepends=True)[:4])
    path = tmp_path / "header.csv"
    path.write_text(header, encoding="utf-8")

    with pytest.raises(ValueError, match="holds no FBRM sample"):
        PATReaders.read_ic_fbrm_export(path)


def test_fbrm_table_reduces_export_samples_to_the_vendor_statistics():
    fbrm = PATReaders.read_ic_fbrm_export(FBRM_EXPORT)

    origin = reduce_runs.run_clock_origin(fbrm.time[0].item(), RUN, DESCRIPTION)
    columns, _ = reduce_runs.fbrm_table(fbrm, origin, DESCRIPTION)

    # The first sample closes a 10 s interval that began 170 s before the log: 10 - 170 s.
    # The others follow 4 h 27 min 1 s and 10 h 0 min 50 s later on the export's whole seconds.
    assert columns["time"].tolist() == [-160.0, -160.0 + 16021.0, -160.0 + 36050.0]
    # Statistics stored by the vendor software for these samples (see test_pat_signal_statistics).
    np.testing.assert_allclose(columns["counts_10_to_100um"], [25075.2, 33528.3, 28549.4], rtol=1e-2)
    np.testing.assert_allclose(columns["mean_chord_square_weighted"], [121.868, 77.0612, 90.7105], rtol=5e-4)


def test_fbrm_table_takes_the_channels_from_the_samples():
    origin = datetime.datetime(2026, 10, 24, 0, 59, 0)
    names = ["counts_1_to_10um", "counts_10_to_100um", "counts_100_to_1000um"]

    # Five unequal channels with edges on the band edges 10 and 100 um: plain sums.
    on_edges, _ = reduce_runs.fbrm_table(_samples([1, 4, 10, 40, 100, 1000], [3, 6, 8, 12, 2]), origin, DESCRIPTION)
    # The channel from 1 to 19 um lies half below 10 um, so its 18 counts split 9 and 9.
    across, _ = reduce_runs.fbrm_table(_samples([1, 19, 100, 1000], [18, 81, 5]), origin, DESCRIPTION)

    assert on_edges["time"].tolist() == [60.0]
    assert [on_edges[name].tolist() for name in names] == [pytest.approx([9.0]), pytest.approx([20.0]), pytest.approx([2.0])]
    assert [across[name].tolist() for name in names] == [pytest.approx([9.0]), pytest.approx([90.0]), pytest.approx([5.0])]


@pytest.mark.parametrize("edges", [[2, 10, 100, 1000], [1, 10, 100, 500]])
def test_fbrm_table_refuses_bands_beyond_the_channels(edges):
    with pytest.raises(ValueError, match="The bands, 1 to 1000 um, reach beyond the channels of the samples, %d to %d um"
                       % (edges[0], edges[-1])):
        reduce_runs.fbrm_table(_samples(edges, [1, 1, 1]), datetime.datetime(2026, 10, 24), DESCRIPTION)


def _export_run(folder: Path) -> tuple[dict, dict]:
    """Fill a run folder with the two exports and the thermostat log, and describe it."""
    folder.mkdir()
    shutil.copy(IR_EXPORT, folder / "ir.csv")
    shutil.copy(FBRM_EXPORT, folder / "fbrm.csv")
    shutil.copy(FIXTURES / "huber_log_excerpt.csv", folder / DESCRIPTION["temperature_log"]["file"])
    description = copy.deepcopy(DESCRIPTION)
    description["temperature_log"]["block_samples"] = 3  # the excerpt has five rows
    run = dict(RUN, folder=folder.name, fbrm_export="fbrm.csv", ir_export="ir.csv", ir_export_start_local=IR_START_LOCAL)
    return run, description


def test_reduce_run_reads_the_exports_a_run_names(tmp_path):
    run, description = _export_run(tmp_path / "run")
    out = tmp_path / "out"

    rows, _ = reduce_runs.reduce_run(CASE, tmp_path, run, description, out)

    assert rows == {"temperature": 1, "fbrm": 3, "ir": 3}
    fbrm, _ = table_layout.read_table(out / "high_seed_slow_cool_fbrm.csv", table_layout.load_layout(CASE / "reduced_fbrm.layout.json"))
    ir, _ = table_layout.read_table(out / "high_seed_slow_cool_ir.csv", table_layout.load_layout(CASE / "reduced_ir.layout.json"))
    assert fbrm["time"].tolist() == [-160.0, 15861.0, 35890.0]
    np.testing.assert_allclose(fbrm["counts_10_to_100um"], [25075.2, 33528.3, 28549.4], rtol=1e-2)
    # Run clock zero is 01:53:23 - 10 s + 170 s = 01:56:03 local. The IR experiment started at
    # 00:28:19 local, so its spectra titled 00:00:51, 00:01:53 and 00:02:51 were taken at
    # 00:29:10, 00:30:12 and 00:31:10: 5213, 5151 and 5093 s before zero.
    assert ir["time"].tolist() == [-5213.0, -5151.0, -5093.0]
    # Export absorbances at 1760, 1720 and 1700 cm-1 of the same spectra; the table keeps five decimals.
    a1760 = np.array([0.146599612457919, 0.147050707666293, 0.148026113384757])
    a1720 = np.array([0.248405187363523, 0.248816188191783, 0.250148186896707])
    a1700 = np.array([0.364370714213183, 0.365671123431154, 0.367121402604299])
    np.testing.assert_allclose(ir["band_height"], a1700 - 1.5 * a1720 + 0.5 * a1760, atol=1e-5)


def test_reduce_run_reads_the_native_file_of_a_signal_without_an_export(tmp_path):
    run, description = _export_run(tmp_path / "run")
    del run["fbrm_export"]
    shutil.copy(FIXTURES / "fbrm_three_samples.icFbrm", tmp_path / "run" / "samples.icFbrm")

    rows, _ = reduce_runs.reduce_run(CASE, tmp_path, run, description, tmp_path / "out")

    fbrm, _ = table_layout.read_table(tmp_path / "out" / "high_seed_slow_cool_fbrm.csv",
                                      table_layout.load_layout(CASE / "reduced_fbrm.layout.json"))
    assert rows["fbrm"] == 3 and zipfile.is_zipfile(tmp_path / "run" / "samples.icFbrm")
    # Native sample times keep their fraction of a second: 16021.612152 s between the first two.
    np.testing.assert_allclose(fbrm["time"], [-160.0, -160.0 + 16021.6, -160.0 + 36049.7], atol=0.05)


@pytest.mark.parametrize("missing", ["ir_export", "ir_export_start_local"])
def test_reduce_run_needs_the_ir_export_and_its_start_together(tmp_path, missing):
    run, description = _export_run(tmp_path / "run")
    del run[missing]

    with pytest.raises(ValueError, match="Run high_seed_slow_cool must give ir_export and ir_export_start_local together"):
        reduce_runs.reduce_run(CASE, tmp_path, run, description, tmp_path / "out")
