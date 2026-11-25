"""Native-file PAT readers of ``PharmaPy.PATReaders`` against the vendor's own exports.

Fixtures in ``tests/fixtures/pat`` are excerpts of run 2026-10-23; its
README states how they were cut and which identifying values were overwritten.
Expected values come from the vendor's CSV exports, not from the readers.
"""
import csv
import datetime
import struct
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from PharmaPy import PATReaders

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "pat"
VECTOR_PART = "/ReactionSpectras.vector"
EXPERIMENT_PART = "/SpectraExperiment0.xml"
# The export prints 15 significant digits of absorbances below 1, so it agrees with
# the stored float64 to about 1e-15; 1e-12 is far below the 2e-4 spectral noise.
ABSORBANCE_TOLERANCE = 1e-12  # [-]
# Largest relative difference between rebinned raw counts and the export over the
# 4313 samples of the run was 3.3e-3 (inferred raw grid); totals agreed to 1e-5.
CHANNEL_TOLERANCE = 5e-3  # [-]
TOTAL_TOLERANCE = 1e-4  # [-]
IR_BUILD, FBRM_BUILD = b"7.1.91.0", b"4.4.33.0"  # builds that wrote the fixtures
RAMAN_ID = "8b460ded-99bc-44c0-8964-96806ad6ee53"  # the iC Raman entry of spectrum_info_tags.xml
RAW_CHANNELS = 1550  # [-] channels of an iC FBRM sample


def _ir_fixture() -> bytes:
    return (FIXTURES / "ir_head_blocks.iCIR").read_bytes()


def _info_tags() -> str:
    return (FIXTURES / "spectrum_info_tags.xml").read_text(encoding="utf-8")


def _export_columns() -> dict:
    with open(FIXTURES / "ir_export_columns.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    table = np.array([[float(value) for value in row] for row in rows[1:]])
    return {name: table[:, column] for column, name in enumerate(rows[0])}


def _complete_ir_file(folder: Path, data: bytes = None) -> Path:
    """Write the IR excerpt with an experiment document, as a whole file has.

    The excerpt keeps four blocks, so its experiment document is missing. Here
    the metadata tags of ``spectrum_info_tags.xml`` are stored as that document
    in a fifth block, and its name-table entry is pointed at the block.
    """
    data = bytearray(_ir_fixture() if data is None else data)
    document = _info_tags().encode("utf-8")
    payload = struct.pack("<I", len(document)) + document
    block_offset = -(-len(data) // PATReaders.BLOCK_SIZE) * PATReaders.BLOCK_SIZE  # next 64 KiB boundary
    (table_offset,) = struct.unpack_from("<Q", data, PATReaders.DIRECTORY_FIELD_OFFSET)
    position = table_offset + 4  # past the uint32 number of parts
    while True:
        name_length = data[position]
        name = bytes(data[position + 1: position + 1 + name_length]).decode("utf-8")
        position += 1 + name_length
        position += 1 + data[position]  # content type
        if name == EXPERIMENT_PART:
            struct.pack_into("<Q", data, position, block_offset)
            break
        position += PATReaders.NAME_ENTRY_SIZE
    data += bytes(block_offset - len(data))
    data += struct.pack("<IIQ", PATReaders.STREAM_HEAD_FLAG, len(payload), 0) + bytes(PATReaders.STREAM_HEAD_RESERVED) + payload
    path = folder / "complete.iCIR"
    path.write_bytes(bytes(data))
    return path


def _fbrm_vector() -> bytes:
    with zipfile.ZipFile(FIXTURES / "fbrm_three_samples.icFbrm") as archive:
        return archive.read(PATReaders.SAMPLES_PART)


def _fbrm_file(folder: Path, vector: bytes, part: str = PATReaders.SAMPLES_PART) -> Path:
    path = folder / "samples.icFbrm"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(part, vector)
    return path


def test_finalized_package_is_read_through_its_name_table():
    parts = PATReaders.read_package(_ir_fixture())

    assert PATReaders.is_finalized(_ir_fixture())
    content_type, stream = parts[VECTOR_PART]
    assert content_type == "application/vnd.mtac-data.ic.SpectraVectorList"
    assert parts["/FileInfo.xml"][1].startswith(b"<iCFileInformation")
    # The fixture keeps four blocks, so parts stored further on are listed but empty.
    assert parts[EXPERIMENT_PART][1] == b""
    assert len(stream) == 63468


def test_unfinalized_package_recovers_the_same_streams():
    data = bytearray(_ir_fixture())
    data[16:28] = bytes(12)  # a file the software never closed has no name table

    recovered = PATReaders.read_package(bytes(data))

    assert not PATReaders.is_finalized(bytes(data))
    assert sorted(recovered) == ["/FileInfo.xml", VECTOR_PART, "/_rels/.rels"]
    assert recovered[VECTOR_PART] == PATReaders.read_package(_ir_fixture())[VECTOR_PART]


def test_read_package_rejects_other_files():
    with pytest.raises(ValueError, match="Not a BlockPackage file"):
        PATReaders.read_package(b"PK\x03\x04" + bytes(64))


def test_ir_spectra_match_the_vendor_export():
    infos = PATReaders.spectrum_infos(_info_tags())
    stream = PATReaders.read_package(_ir_fixture())[VECTOR_PART][1]
    records = dict(PATReaders.spectrum_records(stream))
    export = _export_columns()

    assert len(records) == 6
    compared = 0
    for identifier, attributes in infos.items():
        if identifier not in records:
            continue
        values = records[identifier]
        assert values.shape == (int(attributes["NumberOfPoints"]),) == (589,)
        np.testing.assert_allclose(values, export[attributes["Name"]], rtol=0.0, atol=ABSORBANCE_TOLERANCE)
        compared += 1
    assert compared == 3
    # The three spectra differ, so a column mix-up would not pass.
    assert np.abs(export["00:00:51"] - export["00:01:53"]).max() > 1e-3
    assert export["Wavenumbercm-1"][[0, -1]].tolist() == [3000.0, 648.0]


def test_spectrum_infos_reads_reference_reaction_and_raman_entries():
    infos = PATReaders.spectrum_infos(_info_tags())

    assert [attributes["Name"] for attributes in infos.values()][:4] == ["Water Background", "00:00:51", "00:01:53", "00:02:51"]
    raman = infos[RAMAN_ID]
    assert (raman["ExposureTime"], raman["NumberOfPoints"], raman["YUnits"]) == ("PT1S", "3251", "Intensity")
    assert infos["547dd5eb-0d44-4dd6-a7b0-ba913d526fea"]["SampleTime"] == "2026-10-24T04:29:10.2574757Z"


def test_read_ic_spectra_returns_the_export_in_order_of_sample_time(tmp_path):
    export = _export_columns()
    names = ("00:00:51", "00:01:53", "00:02:51")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the fixture was written by a checked build
        ir = PATReaders.read_ic_spectra(_complete_ir_file(tmp_path))

    assert ir.names == names
    # Sample times of the three metadata entries, UTC, truncated to microseconds.
    assert ir.time.dtype == np.dtype("datetime64[us]")
    assert ir.time.tolist() == [datetime.datetime(2026, 10, 24, 4, 29, 10, 257475),
                                datetime.datetime(2026, 10, 24, 4, 30, 11, 952319),
                                datetime.datetime(2026, 10, 24, 4, 31, 10, 578463)]
    np.testing.assert_allclose(ir.wavenumber, export["Wavenumbercm-1"], rtol=0.0, atol=1e-9)
    assert ir.absorbance.shape == (3, 589)
    for row, name in enumerate(names):
        np.testing.assert_allclose(ir.absorbance[row], export[name], rtol=0.0, atol=ABSORBANCE_TOLERANCE)
    # The excerpt holds six spectra; three have no metadata entry, so no sample time.
    assert len(ir.without_time) == 3 and not set(ir.without_time) & set(PATReaders.spectrum_infos(_info_tags()))
    assert (ir.value_units, ir.software, ir.build) == ("Absorbance", "iC IR", IR_BUILD.decode())


def test_spectra_are_sorted_by_sample_time_whatever_the_record_order():
    infos = PATReaders.spectrum_infos(_info_tags())
    stream = PATReaders.read_package(_ir_fixture())[VECTOR_PART][1]
    records = dict(PATReaders.spectrum_records(stream))
    reversed_records = {identifier: records[identifier] for identifier in reversed(list(records))}
    export = _export_columns()

    ir = PATReaders.spectra_from_records(infos, reversed_records)

    assert ir.names == ("00:00:51", "00:01:53", "00:02:51") and np.all(np.diff(ir.time) > np.timedelta64(0, "us"))
    np.testing.assert_allclose(ir.values[2], export["00:02:51"], rtol=0.0, atol=ABSORBANCE_TOLERANCE)
    assert (ir.software, ir.build) == ("", "")


def test_sample_time_without_fraction_is_parsed():
    identifier = "547dd5eb-0d44-4dd6-a7b0-ba913d526fea"
    infos = PATReaders.spectrum_infos(_info_tags().replace("2026-10-24T04:29:10.2574757Z", "2026-10-24T04:29:10Z"))
    stream = PATReaders.read_package(_ir_fixture())[VECTOR_PART][1]
    records = dict(PATReaders.spectrum_records(stream))

    ir = PATReaders.spectra_from_records(infos, {identifier: records[identifier]})

    assert ir.time.tolist() == [datetime.datetime(2026, 10, 24, 4, 29, 10)]


def test_read_ic_spectra_warns_for_a_build_that_was_not_checked(tmp_path):
    other = _ir_fixture().replace(IR_BUILD, b"7.2.91.0")

    with pytest.warns(UserWarning, match=r"written by iC IR build 7\.2\.91\.0\. This reader was checked .* iC IR 7\.1\.91\.0"):
        ir = PATReaders.read_ic_spectra(_complete_ir_file(tmp_path, other))

    assert ir.build == "7.2.91.0" and ir.absorbance.shape == (3, 589)


def test_read_ic_spectra_warns_when_only_the_last_writer_was_not_checked(tmp_path):
    data = _ir_fixture()
    updated = data.index(IR_BUILD, data.index(b"m_UpdatedWith"))
    mixed = data[:updated] + b"7.2.91.0" + data[updated + len(IR_BUILD):]

    with pytest.warns(UserWarning, match=r"iC IR build 7\.1\.91\.0, 7\.2\.91\.0"):
        ir = PATReaders.read_ic_spectra(_complete_ir_file(tmp_path, mixed))

    assert ir.build == IR_BUILD.decode()


def test_read_ic_spectra_refuses_a_file_without_its_experiment_document(tmp_path):
    truncated = tmp_path / "truncated.iCIR"
    truncated.write_bytes(_ir_fixture())  # the experiment document lies beyond the four blocks kept
    unfinalized = bytearray(_ir_fixture())
    unfinalized[16:28] = bytes(12)  # no name table; the scan finds no experiment document in four blocks
    recovered = tmp_path / "recovered.iCIR"
    recovered.write_bytes(bytes(unfinalized))

    with pytest.raises(ValueError, match="experiment document of .* is empty"):
        PATReaders.read_ic_spectra(truncated)
    with pytest.raises(ValueError, match=r"Expected one /SpectraExperiment\*\.xml part in .*, found 0"):
        PATReaders.read_ic_spectra(recovered)


def test_raman_spectra_are_not_handed_over_as_absorbance():
    infos = PATReaders.spectrum_infos(_info_tags())
    intensity = np.arange(3251.0)  # one value per point of the Raman entry

    raman = PATReaders.spectra_from_records(infos, {RAMAN_ID: intensity})

    assert raman.value_units == "Intensity" and raman.wavenumber[[0, -1]].tolist() == [3400.0, 150.0]
    np.testing.assert_array_equal(raman.values, intensity[None, :])
    with pytest.raises(ValueError, match="hold 'Intensity', not absorbance"):
        raman.absorbance


def test_spectra_from_records_refuses_records_it_cannot_join():
    infos = PATReaders.spectrum_infos(_info_tags())
    stream = PATReaders.read_package(_ir_fixture())[VECTOR_PART][1]
    records = dict(PATReaders.spectrum_records(stream))
    first, second = [identifier for identifier in records if identifier in infos][:2]
    other_units = {**infos, second: {**infos[second], "YUnits": "Transmittance"}}
    no_time = {**infos, first: {key: value for key, value in infos[first].items() if key != "SampleTime"}}

    with pytest.raises(ValueError, match="do not share one spectral axis"):
        PATReaders.spectra_from_records(infos, {first: records[first], RAMAN_ID: np.zeros(3251)})
    with pytest.raises(ValueError, match="do not share one ordinate"):
        PATReaders.spectra_from_records(other_units, {first: records[first], second: records[second]})
    with pytest.raises(ValueError, match="holds 588 values but its metadata entry declares 589 points"):
        PATReaders.spectra_from_records(infos, {first: records[first][:-1]})
    with pytest.raises(ValueError, match="lacks SampleTime"):
        PATReaders.spectra_from_records(no_time, {first: records[first]})
    with pytest.raises(ValueError, match="No spectrum has a metadata entry"):
        PATReaders.spectra_from_records({}, records)


def test_fbrm_samples_match_the_vendor_export():
    with open(FIXTURES / "fbrm_export_rows.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    export_edges = np.array([float(value) for value in rows[2][4:105]])  # [um]
    export_counts = np.array([[float(value) for value in row[4:104]] for row in rows[4:]])  # [counts]
    export_times = [datetime.datetime.strptime(row[1] + " " + row[2], "%m/%d/%Y %I:%M:%S %p") for row in rows[4:]]

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the fixture was written by a checked build
        fbrm = PATReaders.read_ic_fbrm(FIXTURES / "fbrm_three_samples.icFbrm")
    rebinned = fbrm.counts @ PATReaders.rebin_matrix(fbrm.channel_edges, export_edges)

    assert fbrm.counts.shape == (3, RAW_CHANNELS) and fbrm.time.dtype == np.dtype("datetime64[us]")
    assert [time.replace(microsecond=0) for time in fbrm.time.tolist()] == export_times
    np.testing.assert_array_equal(fbrm.channel_edges, PATReaders.raw_channel_edges())
    assert (fbrm.software, fbrm.build) == ("iC FBRM", FBRM_BUILD.decode())
    scale = np.maximum(export_counts, 1.0)
    assert (np.abs(rebinned - export_counts) / scale).max() < CHANNEL_TOLERANCE
    np.testing.assert_allclose(rebinned.sum(axis=1), export_counts.sum(axis=1), rtol=TOTAL_TOLERANCE)
    # The three samples differ by thousands of counts, so a row mix-up would not pass.
    assert export_counts.sum(axis=1).tolist() == pytest.approx([36776.6, 51983.5, 40899.8], abs=0.1)


def test_fbrm_reader_stops_at_the_recorded_end_of_records(tmp_path):
    """The offset at byte 164 bounds the records; the trailing file map is not a sample."""
    vector = bytearray(_fbrm_vector())
    record_size = (len(vector) - 5478) // 3
    struct.pack_into("<I", vector, PATReaders.RECORDS_END_FIELD, len(vector) - record_size)

    fbrm = PATReaders.read_ic_fbrm(_fbrm_file(tmp_path, bytes(vector)))

    assert fbrm.time.shape == (2,) and fbrm.counts.shape == (2, RAW_CHANNELS)


def test_read_ic_fbrm_warns_for_a_build_that_was_not_checked(tmp_path):
    other = _fbrm_vector().replace(b"Version=" + FBRM_BUILD, b"Version=4.5.33.0")

    with pytest.warns(UserWarning, match=r"written by iC FBRM build 4\.5\.33\.0"):
        fbrm = PATReaders.read_ic_fbrm(_fbrm_file(tmp_path, other))

    assert fbrm.build == "4.5.33.0" and fbrm.counts.shape == (3, RAW_CHANNELS)


def test_read_ic_fbrm_refuses_files_without_usable_samples(tmp_path):
    vector = _fbrm_vector()
    header_only = vector[:5478]  # the stream of a file that recorded no sample ends before the first record
    array_header = struct.pack("<i", RAW_CHANNELS) + bytes([PATReaders.DOUBLE_TYPE])
    shorter = vector.replace(array_header, struct.pack("<i", RAW_CHANNELS - 1) + bytes([PATReaders.DOUBLE_TYPE]), 1)

    with pytest.raises(ValueError, match="No FBRM sample record"):
        PATReaders.read_ic_fbrm(_fbrm_file(tmp_path, header_only))
    with pytest.raises(ValueError, match="holds no ReactionSamples.vector part"):
        PATReaders.read_ic_fbrm(_fbrm_file(tmp_path, vector, part="Statistics"))
    with pytest.raises(ValueError, match=r"samples of \[1549, 1550\] channels; the raw grid of this reader has 1550"):
        PATReaders.read_ic_fbrm(_fbrm_file(tmp_path, shorter))


def test_raw_channel_edges_cover_the_three_resolutions():
    edges = PATReaders.raw_channel_edges()

    assert edges.shape == (1551,)
    assert edges[[0, 500, 900, 1550]].tolist() == pytest.approx([0.0, 100.0, 500.0, 1800.0])
    assert np.diff(edges)[[0, 499, 500, 899, 900, 1549]].tolist() == pytest.approx([0.2, 0.2, 1.0, 1.0, 2.0, 2.0])


def test_rebin_matrix_splits_counts_by_overlapping_length():
    source_edges = np.array([0.0, 2.0, 4.0, 10.0])  # [um] three channels
    target_edges = np.array([0.0, 1.0, 7.0])  # [um] two channels; 7 to 10 um is outside

    matrix = PATReaders.rebin_matrix(source_edges, target_edges)

    np.testing.assert_allclose(matrix, [[0.5, 0.5], [0.0, 1.0], [0.0, 0.5]])
    np.testing.assert_allclose(np.array([8.0, 3.0, 6.0]) @ matrix, [4.0, 10.0])


@pytest.mark.parametrize("edges", [np.array([0.0, 1.0, 1.0]), np.array([[0.0, 1.0]]), np.array([2.0])])
def test_rebin_matrix_rejects_invalid_edges(edges):
    with pytest.raises(ValueError, match="strictly increasing"):
        PATReaders.rebin_matrix(edges, np.array([0.0, 1.0]))
