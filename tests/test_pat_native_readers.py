"""Native-file PAT readers in ``examples/pat`` against the vendor's own exports.

Fixtures in ``tests/fixtures/pat`` are excerpts of run 2026-10-23; its
README states how they were cut and which identifying values were overwritten.
Expected values come from the vendor's CSV exports, not from the readers.
"""
import csv
import datetime
import io
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from examples.pat import block_package, ic_fbrm, ic_spectra

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "pat"
VECTOR_PART = "/ReactionSpectras.vector"
# The export prints 15 significant digits of absorbances below 1, so it agrees with
# the stored float64 to about 1e-15; 1e-12 is far below the 2e-4 spectral noise.
ABSORBANCE_TOLERANCE = 1e-12  # [-]
# Largest relative difference between rebinned raw counts and the export over the
# 4313 samples of the run was 3.3e-3 (inferred raw grid); totals agreed to 1e-5.
CHANNEL_TOLERANCE = 5e-3  # [-]
TOTAL_TOLERANCE = 1e-4  # [-]


def _ir_fixture() -> bytes:
    return (FIXTURES / "ir_head_blocks.iCIR").read_bytes()


def _export_columns() -> dict:
    with open(FIXTURES / "ir_export_columns.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    table = np.array([[float(value) for value in row] for row in rows[1:]])
    return {name: table[:, column] for column, name in enumerate(rows[0])}


def test_finalized_package_is_read_through_its_name_table():
    parts = block_package.read_package(_ir_fixture())

    assert block_package.is_finalized(_ir_fixture())
    content_type, stream = parts[VECTOR_PART]
    assert content_type == "application/vnd.mtac-data.ic.SpectraVectorList"
    assert parts["/FileInfo.xml"][1].startswith(b"<iCFileInformation")
    # The fixture keeps four blocks, so parts stored further on are listed but empty.
    assert parts["/SpectraExperiment0.xml"][1] == b""
    assert len(stream) == 63468


def test_unfinalized_package_recovers_the_same_streams():
    data = bytearray(_ir_fixture())
    data[16:28] = bytes(12)  # a file the software never closed has no name table

    recovered = block_package.read_package(bytes(data))

    assert not block_package.is_finalized(bytes(data))
    assert sorted(recovered) == ["/FileInfo.xml", VECTOR_PART, "/_rels/.rels"]
    assert recovered[VECTOR_PART] == block_package.read_package(_ir_fixture())[VECTOR_PART]


def test_read_package_rejects_other_files():
    with pytest.raises(ValueError, match="Not a BlockPackage file"):
        block_package.read_package(b"PK\x03\x04" + bytes(64))


def test_ir_spectra_match_the_vendor_export():
    infos = ic_spectra.spectrum_infos((FIXTURES / "spectrum_info_tags.xml").read_text(encoding="utf-8"))
    stream = block_package.read_package(_ir_fixture())[VECTOR_PART][1]
    records = dict(ic_spectra.spectrum_records(stream))
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
    infos = ic_spectra.spectrum_infos((FIXTURES / "spectrum_info_tags.xml").read_text(encoding="utf-8"))

    assert [attributes["Name"] for attributes in infos.values()][:4] == ["Water Background", "00:00:51", "00:01:53", "00:02:51"]
    raman = infos["8b460ded-99bc-44c0-8964-96806ad6ee53"]
    assert (raman["ExposureTime"], raman["NumberOfPoints"], raman["YUnits"]) == ("PT1S", "3251", "Intensity")
    assert infos["547dd5eb-0d44-4dd6-a7b0-ba913d526fea"]["SampleTime"] == "2026-10-24T04:29:10.2574757Z"


def test_fbrm_samples_match_the_vendor_export():
    with open(FIXTURES / "fbrm_export_rows.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    export_edges = np.array([float(value) for value in rows[2][4:105]])  # [um]
    export_counts = np.array([[float(value) for value in row[4:104]] for row in rows[4:]])  # [counts]
    export_times = [datetime.datetime.strptime(row[1] + " " + row[2], "%m/%d/%Y %I:%M:%S %p") for row in rows[4:]]

    times, raw = ic_fbrm.read_samples(FIXTURES / "fbrm_three_samples.icFbrm")
    rebinned = raw @ ic_fbrm.rebin_matrix(ic_fbrm.raw_channel_edges(), export_edges)

    assert raw.shape == (3, 1550)
    assert [time.replace(microsecond=0) for time in times] == export_times
    scale = np.maximum(export_counts, 1.0)
    assert (np.abs(rebinned - export_counts) / scale).max() < CHANNEL_TOLERANCE
    np.testing.assert_allclose(rebinned.sum(axis=1), export_counts.sum(axis=1), rtol=TOTAL_TOLERANCE)
    # The three samples differ by thousands of counts, so a row mix-up would not pass.
    assert export_counts.sum(axis=1).tolist() == pytest.approx([36776.6, 51983.5, 40899.8], abs=0.1)


def test_fbrm_reader_stops_at_the_recorded_end_of_records():
    """The offset at byte 164 bounds the records; the trailing file map is not a sample."""
    with zipfile.ZipFile(FIXTURES / "fbrm_three_samples.icFbrm") as archive:
        vector = bytearray(archive.read(ic_fbrm.SAMPLES_PART))
    record_size = (len(vector) - 5478) // 3
    struct.pack_into("<I", vector, ic_fbrm.RECORDS_END_FIELD, len(vector) - record_size)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(ic_fbrm.SAMPLES_PART, bytes(vector))

    times, raw = ic_fbrm.read_samples(buffer)

    assert len(times) == 2 and raw.shape == (2, 1550)


def test_raw_channel_edges_cover_the_three_resolutions():
    edges = ic_fbrm.raw_channel_edges()

    assert edges.shape == (1551,)
    assert edges[[0, 500, 900, 1550]].tolist() == pytest.approx([0.0, 100.0, 500.0, 1800.0])
    assert np.diff(edges)[[0, 499, 500, 899, 900, 1549]].tolist() == pytest.approx([0.2, 0.2, 1.0, 1.0, 2.0, 2.0])


def test_rebin_matrix_splits_counts_by_overlapping_length():
    source_edges = np.array([0.0, 2.0, 4.0, 10.0])  # [um] three channels
    target_edges = np.array([0.0, 1.0, 7.0])  # [um] two channels; 7 to 10 um is outside

    matrix = ic_fbrm.rebin_matrix(source_edges, target_edges)

    np.testing.assert_allclose(matrix, [[0.5, 0.5], [0.0, 1.0], [0.0, 0.5]])
    np.testing.assert_allclose(np.array([8.0, 3.0, 6.0]) @ matrix, [4.0, 10.0])


@pytest.mark.parametrize("edges", [np.array([0.0, 1.0, 1.0]), np.array([[0.0, 1.0]]), np.array([2.0])])
def test_rebin_matrix_rejects_invalid_edges(edges):
    with pytest.raises(ValueError, match="strictly increasing"):
        ic_fbrm.rebin_matrix(edges, np.array([0.0, 1.0]))
