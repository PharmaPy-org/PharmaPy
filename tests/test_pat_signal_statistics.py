"""FBRM statistics, IR band height and the table writer of ``examples/pat``.

Vendor expectations are the statistics iC FBRM stored for samples 401, 2001 and
4001 of run 2026-10-23, and absorbances printed in the iC IR export of that run.
"""
import csv
from pathlib import Path

import numpy as np
import pytest

from examples.pat import block_package, fbrm_statistics, ic_fbrm, ic_spectra, table_layout
from examples.pat.band_height import band_height

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "pat"
# Values the vendor software stored in the trends of the native file.
STORED_BAND_COUNTS = np.array([[10300.4, 25075.2, 1400.93], [17855.3, 33528.3, 600.378], [11408.1, 28549.4, 942.453]])  # [counts]
STORED_SQUARE_WEIGHTED_MEAN = np.array([121.868, 77.0612, 90.7105])  # [um]
# Largest relative differences found over all samples of the four caffeine runs: 6e-3
# for 10 to 100 um, 3e-2 above 100 um (the raw grid there is inferred), 2.2e-4 for the
# mean. The band below 10 um differs by 5e-4 on these three samples.
BAND_TOLERANCE = np.array([1e-2, 1e-2, 4e-2])  # [-]
MEAN_TOLERANCE = 5e-4  # [-]


def test_band_counts_and_mean_chord_match_the_statistics_stored_by_the_vendor():
    _, raw = ic_fbrm.read_samples(FIXTURES / "fbrm_three_samples.icFbrm")
    raw_edges = ic_fbrm.raw_channel_edges()
    log_edges = fbrm_statistics.log_channel_edges()

    bands = fbrm_statistics.band_counts(raw, raw_edges, np.array([1.0, 10.0, 100.0, 1000.0]))
    log_counts = fbrm_statistics.band_counts(raw, raw_edges, log_edges)
    mean = fbrm_statistics.weighted_mean_chord(log_counts, fbrm_statistics.geometric_midpoints(log_edges))

    assert np.all(np.abs(bands - STORED_BAND_COUNTS) / STORED_BAND_COUNTS < BAND_TOLERANCE)
    np.testing.assert_allclose(mean, STORED_SQUARE_WEIGHTED_MEAN, rtol=MEAN_TOLERANCE)


def test_log_channel_grid_matches_the_export_header():
    with open(FIXTURES / "fbrm_export_rows.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    export_edges = np.array([float(value) for value in rows[2][4:105]])  # [um]
    export_midpoints = np.array([float(value) for value in rows[3][4:104]])  # [um]

    edges = fbrm_statistics.log_channel_edges()

    # The export prints five significant digits.
    np.testing.assert_allclose(edges, export_edges, rtol=1e-4)
    np.testing.assert_allclose(fbrm_statistics.geometric_midpoints(edges), export_midpoints, rtol=1e-4)
    assert edges.shape == (101,)


def test_log_channel_edges_rejects_invalid_limits():
    with pytest.raises(ValueError, match="0 < lower < upper"):
        fbrm_statistics.log_channel_edges(lower=0.0)


def test_weighted_mean_chord_follows_its_definition():
    counts = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 2.0]])  # [counts]
    midpoints = np.array([10.0, 20.0, 40.0])  # [um]

    square = fbrm_statistics.weighted_mean_chord(counts, midpoints)
    unweighted = fbrm_statistics.weighted_mean_chord(counts, midpoints, weight_power=0.0)

    # Row 0: (1*10**3 + 3*20**3) / (1*10**2 + 3*20**2) = 25000 / 1300.
    # Row 2: (2*10**3 + 2*40**3) / (2*10**2 + 2*40**2) = 130000 / 3400.
    np.testing.assert_allclose(square[[0, 2]], [25000.0 / 1300.0, 130000.0 / 3400.0])
    np.testing.assert_allclose(unweighted[[0, 2]], [17.5, 25.0])
    assert np.isnan(square[1]) and np.isnan(unweighted[1])


def test_band_height_removes_a_sloped_baseline_on_a_descending_axis():
    axis = np.arange(1800.0, 1596.0, -4.0)  # [cm-1] descending, as iC IR stores it
    baseline = 0.2 + 5e-4 * (axis - 1600.0)  # [-] straight line with a slope
    band = np.where(axis == 1700.0, 0.07, 0.0)  # [-] 0.07 above the line at 1700 cm-1 only
    spectra = np.vstack([baseline + band, baseline + 2.0 * band, baseline])

    two_sided = band_height(axis, spectra, peak=1700.0, baseline=(1760.0, 1640.0))
    one_sided = band_height(axis, spectra, peak=1700.0, baseline=(1760.0, 1720.0))

    np.testing.assert_allclose(two_sided, [0.07, 0.14, 0.0], atol=1e-15)
    np.testing.assert_allclose(one_sided, [0.07, 0.14, 0.0], atol=1e-15)
    assert band_height(axis, spectra[0], 1700.0, (1760.0, 1720.0)).shape == ()


def test_band_height_of_native_spectra_matches_the_vendor_export():
    """Reader to band height: 1700 cm-1 above the line through 1760 and 1720 cm-1."""
    infos = ic_spectra.spectrum_infos((FIXTURES / "spectrum_info_tags.xml").read_text(encoding="utf-8"))
    stream = block_package.read_package((FIXTURES / "ir_head_blocks.iCIR").read_bytes())["/ReactionSpectras.vector"][1]
    records = dict(ic_spectra.spectrum_records(stream))
    names = ["00:00:51", "00:01:53", "00:02:51"]
    by_name = {infos[identifier]["Name"]: records[identifier] for identifier in records if identifier in infos}
    attributes = next(a for a in infos.values() if a["Name"] == names[0])
    axis = np.linspace(float(attributes["StartWn"]), float(attributes["EndWn"]), int(attributes["NumberOfPoints"]))  # [cm-1]

    heights = band_height(axis, np.vstack([by_name[name] for name in names]), peak=1700.0, baseline=(1760.0, 1720.0))

    # Export absorbances at 1760, 1720 and 1700 cm-1. The line through the first two,
    # extended to 1700 cm-1, is A1760 + 1.5 (A1720 - A1760).
    a1760 = np.array([0.146599612457919, 0.147050707666293, 0.148026113384757])
    a1720 = np.array([0.248405187363523, 0.248816188191783, 0.250148186896707])
    a1700 = np.array([0.364370714213183, 0.365671123431154, 0.367121402604299])
    np.testing.assert_allclose(heights, a1700 - 1.5 * a1720 + 0.5 * a1760, rtol=0.0, atol=1e-12)
    assert axis[1] - axis[0] == pytest.approx(-4.0)


@pytest.mark.parametrize(
    "peak, baseline, message",
    [
        (500.0, (1760.0, 1720.0), "outside the spectral axis"),
        (1700.0, (1760.0, 1761.0), "fall on the same axis point"),
    ],
)
def test_band_height_rejects_unusable_positions(peak, baseline, message):
    axis = np.arange(1800.0, 1596.0, -4.0)  # [cm-1]

    with pytest.raises(ValueError, match=message):
        band_height(axis, np.zeros(axis.size), peak, baseline)


def test_band_height_rejects_a_spectrum_of_the_wrong_length():
    with pytest.raises(ValueError, match="must match the spectral axis"):
        band_height(np.arange(1800.0, 1596.0, -4.0), np.zeros(7), 1700.0, (1760.0, 1720.0))


def test_written_table_reads_back_through_a_layout(tmp_path):
    path = tmp_path / "reduced.csv"
    columns = {"counts": np.array([12.34, 56.78]), "time": np.array([0.5, 1.5]), "temperature": np.array([293.154, 310.15])}
    units = {"counts": "-", "time": "min", "temperature": "K"}

    table_layout.write_table(path, columns, units, digits={"counts": 1, "time": 1, "temperature": 2})
    layout = table_layout.TableLayout(
        columns=(table_layout.ColumnLayout("time", "time", "min", "time"),
                 table_layout.ColumnLayout("temperature", "temperature", "K"),
                 table_layout.ColumnLayout("counts", "counts", "-")),
        units_row=1, first_data_row=2)
    values, read_units = table_layout.read_table(path, layout)

    assert path.read_text(encoding="utf-8").splitlines()[:3] == ["counts,time,temperature", "-,min,K", "12.3,0.5,293.15"]
    np.testing.assert_allclose(values["time"], [30.0, 90.0])  # [s]
    np.testing.assert_allclose(values["temperature"], [293.15, 310.15])  # [K]
    np.testing.assert_allclose(values["counts"], [12.3, 56.8])
    assert read_units == {"time": "s", "temperature": "K", "counts": "-"}


@pytest.mark.parametrize(
    "columns, message",
    [
        ({"a": np.array([1.0, 2.0]), "b": np.array([1.0])}, "equal length"),
        ({"a": np.array([1.0, np.nan])}, "non-finite"),
        ({"a": np.array([1.0]), "c": np.array([1.0])}, "needs a unit and a number of decimals"),
    ],
)
def test_write_table_refuses_incomplete_columns(tmp_path, columns, message):
    with pytest.raises(ValueError, match=message):
        table_layout.write_table(tmp_path / "bad.csv", columns, units={"a": "-", "b": "-"}, digits={"a": 1, "b": 1})
