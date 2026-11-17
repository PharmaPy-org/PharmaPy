"""Band-height calibration: the relation, the dataset workflow, and the caffeine runs.

The workflow is exercised on a small dataset written by the test, with another
solute, solvent, table and column names than the caffeine case, built from known
constants. The caffeine cases compare against values read from the reduced tables
with a separate script and against hand arithmetic on them.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from examples.pat import band_calibration, calibrate_band, dataset, table_layout

pytestmark = pytest.mark.unit

CAFFEINE = Path(calibrate_band.__file__).parent / "caffeine"
# Hand-built calibration: baseline [-], slope and its temperature coefficient
# [-] and [1/K] per [kg/kg], reference temperature [K].
BASELINE, SLOPE, SLOPE_PER_KELVIN, REFERENCE = 0.05, 2.0, -0.01, 300.0
RATIO = 0.04  # [kg/kg] solute per solvent on both hand-built plateaus
# Slope is 1.9 at 310 K and 2.05 at 295 K, so the plateaus read 0.05 + 0.076 and 0.05 + 0.082.
PLATEAU_HEIGHTS = [0.126, 0.132]  # [-]
PLATEAU_TEMPERATURES = [310.0, 295.0]  # [K]
BAND_TOLERANCE = 1e-4  # [-] the reduced tables store band heights to about this
REFERENCE_RUN = "high_seed_slow_cool"


def test_window_statistics_uses_the_samples_inside_the_window_limits_included():
    time = np.array([0.0, 10.0, 20.0, 30.0, 40.0])  # [s]
    values = np.array([5.0, 1.0, 2.0, 9.0, 7.0])

    median, scatter, count = band_calibration.window_statistics(time, values, 10.0, 30.0)

    assert (median, count) == (2.0, 3)
    assert scatter == pytest.approx(np.sqrt(38.0 / 3.0))  # deviations -3, -2, 5 from the mean 4
    with pytest.raises(ValueError, match="No sample between 41 and 50 s"):
        band_calibration.window_statistics(time, values, 41.0, 50.0)


def test_slope_and_its_temperature_coefficient_follow_from_two_plateaus():
    slope, slope_per_kelvin = band_calibration.fit_slope(PLATEAU_HEIGHTS, PLATEAU_TEMPERATURES, RATIO, BASELINE, REFERENCE)

    assert slope == pytest.approx(SLOPE)
    assert slope_per_kelvin == pytest.approx(SLOPE_PER_KELVIN)


def test_band_height_and_ratio_convert_into_each_other_at_each_temperature():
    ratios = np.array([0.01, 0.03])  # [kg/kg]
    temperatures = np.array([290.0, 310.0])  # [K]; slopes 2.1 and 1.9

    heights = band_calibration.band_height_from_ratio(ratios, temperatures, BASELINE, SLOPE, SLOPE_PER_KELVIN, REFERENCE)

    np.testing.assert_allclose(heights, [0.071, 0.107])
    np.testing.assert_allclose(band_calibration.ratio_from_band_height(
        heights, temperatures, BASELINE, SLOPE, SLOPE_PER_KELVIN, REFERENCE), ratios)


def test_baseline_absorbs_an_offset_of_plateaus_with_a_known_slope():
    shifted = [height + 0.003 for height in PLATEAU_HEIGHTS]

    baseline = band_calibration.fit_baseline(shifted, PLATEAU_TEMPERATURES, RATIO, SLOPE, SLOPE_PER_KELVIN, REFERENCE)

    assert baseline == pytest.approx(0.053)


def test_calibration_refuses_inputs_it_cannot_solve():
    with pytest.raises(ValueError, match="same temperature"):
        band_calibration.fit_slope(PLATEAU_HEIGHTS, [310.0, 310.0], RATIO, BASELINE, REFERENCE)
    with pytest.raises(ValueError, match="Exactly two plateaus"):
        band_calibration.fit_slope([0.1, 0.2, 0.3], [300.0, 310.0, 320.0], RATIO, BASELINE, REFERENCE)
    with pytest.raises(ValueError, match="must be positive"):
        band_calibration.fit_slope(PLATEAU_HEIGHTS, PLATEAU_TEMPERATURES, 0.0, BASELINE, REFERENCE)
    with pytest.raises(ValueError, match="not positive at 500 K"):  # the slope reaches zero 200 K above the reference
        band_calibration.ratio_from_band_height(0.1, 500.0, BASELINE, SLOPE, SLOPE_PER_KELVIN, REFERENCE)
    with pytest.raises(ValueError, match="one-dimensional"):
        band_calibration.fit_baseline(PLATEAU_HEIGHTS, [310.0], RATIO, SLOPE, SLOPE_PER_KELVIN, REFERENCE)
    with pytest.raises(ValueError, match="must be positive"):
        dataset.charge_ratio({"charge": {"solute_mass_g": 10.0, "solvent_mass_g": 0.0}})


# A dataset of another compound. True constants: baseline 0.02 [-], slope 3.0 and
# -0.02 per K at 300 K, so the slope is 3.2 at 290 K, 2.6 at 320 K and 2.9 at 305 K.
# The charge is 10 g solute in 100 g solvent (0.1 kg/kg). The slurry holds 0.03
# kg/kg at the start and 0.035 at the end, both at 290 K. Run "run_b" has the same
# slope on a baseline 0.005 higher, and no solvent-only spectra.
OTHER_SPECTRUM_TIMES = [200.0, 400.0, 600.0, 1300.0, 1500.0, 1700.0, 2300.0, 2500.0, 2700.0, 3300.0, 3500.0, 3700.0]  # [s]
OTHER_HEIGHTS = [0.116] * 3 + [0.28] * 3 + [0.31] * 3 + [0.132] * 3  # [-]
OTHER_RATIOS = [0.03] * 3 + [0.1] * 6 + [0.035] * 3  # [kg/kg]
OTHER_SOLVENT_TIMES = [-300.0, -200.0, -100.0]  # [s], before the temperature log
OTHER_SOLVENT_HEIGHTS = [0.019, 0.020, 0.023]  # [-], median 0.020 and mean 0.0207


@pytest.fixture()
def other_dataset(tmp_path):
    log_time = np.arange(0.0, 4001.0, 100.0)  # [s]
    log_temperature = np.select([log_time < 1000.0, log_time < 2000.0, log_time < 3000.0], [290.0, 320.0, 305.0], 290.0)  # [K]
    (tmp_path / "reduced").mkdir()
    for kind, column, unit in (("uv", "peak", "-"), ("thermo", "liquid_temperature", "K")):
        (tmp_path / ("reduced_%s.layout.json" % kind)).write_text(json.dumps({
            "header_row": 0, "units_row": 1, "first_data_row": 2,
            "columns": {"time": {"header": "time", "unit": "s", "role": "time"}, column: {"header": column, "unit": unit}}}))
    spectra = {"run_a": (OTHER_SOLVENT_TIMES + OTHER_SPECTRUM_TIMES, OTHER_SOLVENT_HEIGHTS + OTHER_HEIGHTS),
               "run_b": (OTHER_SPECTRUM_TIMES, [height + 0.005 for height in OTHER_HEIGHTS])}
    for run_id, (time, height) in spectra.items():
        table_layout.write_table(tmp_path / "reduced" / ("%s_uv.csv" % run_id), {"time": time, "peak": height},
                                 {"time": "s", "peak": "-"}, {"time": 1, "peak": 6})
        table_layout.write_table(tmp_path / "reduced" / ("%s_thermo.csv" % run_id),
                                 {"time": log_time, "liquid_temperature": log_temperature},
                                 {"time": "s", "liquid_temperature": "K"}, {"time": 1, "liquid_temperature": 3})
    (tmp_path / "runs.json").write_text(json.dumps({
        "charge": {"solute": "solute_a", "solvent": "ethanol", "solute_mass_g": 10.0, "solvent_mass_g": 100.0},
        "band_calibration": {
            "table": "uv", "height_column": "peak", "temperature_table": "thermo",
            "temperature_column": "liquid_temperature", "runs": ["run_a", "run_b"], "reference_run": "run_a",
            "reference_temperature_K": 300.0, "solvent_window_s": [-350.0, -50.0],
            "dissolved_windows_s": [[1200.0, 1800.0], [2200.0, 2800.0]], "slurry_start_window_s": [100.0, 700.0],
            "slurry_end_duration_s": 500.0}}))
    return tmp_path


def test_workflow_recovers_the_constants_of_a_dataset_of_another_compound(other_dataset):
    result = calibrate_band.calibrate(other_dataset)

    assert (result["solute"], result["solvent"], result["charge_ratio"]) == ("solute_a", "ethanol", 0.1)
    assert result["solvent_baseline"] == pytest.approx({"band_height": 0.020, "minimum": 0.019, "maximum": 0.023, "count": 3})
    for treatment in calibrate_band.TREATMENTS:
        assert result["runs"]["run_a"][treatment] == pytest.approx({
            "baseline": 0.02, "slope": 3.0, "slope_per_kelvin": -0.02, "slurry_start_ratio": 0.03, "slurry_end_ratio": 0.035})
    assert result["runs"]["run_b"]["common_slope"] == pytest.approx({
        "baseline": 0.025, "slope": 3.0, "slope_per_kelvin": -0.02, "slurry_start_ratio": 0.03, "slurry_end_ratio": 0.035})
    # On the reference baseline the shifted plateaus give slopes 2.65 at 320 K and
    # 2.95 at 305 K: 3.05 at 300 K, 3.25 at 290 K, where 0.101 and 0.117 are measured.
    assert result["runs"]["run_b"]["common_baseline"] == pytest.approx({
        "baseline": 0.02, "slope": 3.05, "slope_per_kelvin": -0.02, "slurry_start_ratio": 0.101 / 3.25,
        "slurry_end_ratio": 0.036})
    plateaus = result["runs"]["run_b"]["plateaus"]
    assert [plateaus[name]["temperature_K"] for name in ("dissolved_first", "dissolved_second", "slurry_start", "slurry_end")] \
        == [320.0, 305.0, 290.0, 290.0]


def test_workflow_writes_the_result_and_converts_a_run_to_solute_per_solvent(other_dataset):
    calibrate_band.main([str(other_dataset)])
    stored = json.loads((other_dataset / calibrate_band.RESULT_NAME).read_text(encoding="utf-8"))

    time, ratio = calibrate_band.solute_ratio(other_dataset, "run_a", "common_baseline", stored)
    np.testing.assert_array_equal(time, OTHER_SPECTRUM_TIMES)  # the solvent spectra have no temperature
    np.testing.assert_allclose(ratio, OTHER_RATIOS, rtol=1e-5)
    np.testing.assert_allclose(calibrate_band.solute_ratio(other_dataset, "run_b", "common_slope", stored)[1], OTHER_RATIOS,
                               rtol=1e-5)
    # The other treatment reads run_b through a different baseline and slope.
    assert calibrate_band.solute_ratio(other_dataset, "run_b", "common_baseline", stored)[1][0] == pytest.approx(0.101 / 3.25, rel=1e-5)
    with pytest.raises(ValueError, match="Run 'run_c' has no band calibration"):
        calibrate_band.solute_ratio(other_dataset, "run_c", "common_baseline", stored)
    with pytest.raises(ValueError, match="Unknown treatment 'per_run'"):
        calibrate_band.solute_ratio(other_dataset, "run_b", "per_run", stored)


@pytest.fixture(scope="module")
def caffeine():
    return calibrate_band.calibrate(CAFFEINE)


def test_caffeine_solvent_baseline_is_the_median_of_the_water_only_spectra(caffeine):
    # The eleven spectra before the caffeine addition read 0.0651, 0.0660, 0.0659,
    # 0.0665, 0.0664, 0.0666, 0.0673, 0.0677, 0.0677, 0.0677 and 0.0681.
    solvent = caffeine["solvent_baseline"]

    assert (caffeine["solute"], caffeine["solvent"], solvent["count"]) == ("caffeine", "water", 11)
    assert solvent["band_height"] == pytest.approx(0.0666, abs=5e-5)
    assert solvent["minimum"] == pytest.approx(0.0651, abs=5e-5)
    assert solvent["maximum"] == pytest.approx(0.0681, abs=5e-5)


@pytest.mark.parametrize("run_id, heights, temperatures_C", [
    ("high_seed_slow_cool", (0.1351, 0.1371, 0.0967, 0.0987), (44.96, 37.18, 20.06, 20.00)),
    ("low_seed_slow_cool", (0.1308, 0.1325, 0.0897, 0.0916), (44.95, 37.20, 20.01, 19.91)),
    ("low_seed_fast_cool", (0.1308, 0.1329, 0.0910, 0.0933), (44.96, 36.79, 20.01, 20.00)),
])
def test_caffeine_plateaus_match_the_reduced_tables(caffeine, run_id, heights, temperatures_C):
    windows = caffeine["runs"][run_id]["plateaus"]
    order = ("dissolved_first", "dissolved_second", "slurry_start", "slurry_end")

    assert [windows[name]["band_height"] for name in order] == pytest.approx(heights, abs=BAND_TOLERANCE)
    assert [windows[name]["temperature_K"] - 273.15 for name in order] == pytest.approx(temperatures_C, abs=0.01)
    assert all(windows[name]["count"] >= 6 for name in order)


def test_caffeine_reference_run_constants_follow_from_its_plateaus_and_the_load(caffeine):
    # Load 16.5434 g in 350 g water. Above the baseline 0.0666 the plateaus read
    # 0.06847 at 318.111 K and 0.07045 at 310.327 K: slopes 1.44858 and 1.49047,
    # so -0.005382 per K and 1.49047 + 0.005382 * 0.177 = 1.49142 at 310.15 K.
    constants = caffeine["runs"][REFERENCE_RUN]["common_baseline"]

    assert caffeine["charge_ratio"] == pytest.approx(0.0472669, rel=1e-6)
    assert constants["slope"] == pytest.approx(1.4914, abs=3e-4)
    assert constants["slope_per_kelvin"] == pytest.approx(-0.00538, abs=2e-5)
    # Start of the run, 20.06 C: 0.03007 above the baseline over a slope of 1.5826.
    assert constants["slurry_start_ratio"] == pytest.approx(0.0190, abs=1e-4)
    assert caffeine["runs"][REFERENCE_RUN]["common_slope"] == constants


def test_caffeine_treatments_split_the_difference_between_runs_as_baseline_or_as_slope(caffeine):
    solvent = caffeine["solvent_baseline"]["band_height"]
    reference = caffeine["runs"][REFERENCE_RUN]["common_baseline"]

    for run_id in ("low_seed_slow_cool", "low_seed_fast_cool"):
        run = caffeine["runs"][run_id]
        assert run["common_baseline"]["baseline"] == solvent
        assert run["common_baseline"]["slope"] == pytest.approx(1.40, abs=0.01)  # 6 % below the reference run
        assert run["common_slope"]["slope"] == reference["slope"]
        assert run["common_slope"]["baseline"] == pytest.approx(0.0623, abs=2e-4)  # plateaus 0.0043 lower


def test_caffeine_slurry_at_20C_reads_higher_at_the_end_of_every_run(caffeine):
    ratios = []  # [kg/kg]
    for run in caffeine["runs"].values():
        for treatment in calibrate_band.TREATMENTS:
            assert run[treatment]["slurry_end_ratio"] - run[treatment]["slurry_start_ratio"] == pytest.approx(0.0013, abs=3e-4)
            ratios += [run[treatment]["slurry_start_ratio"], run[treatment]["slurry_end_ratio"]]

    assert min(ratios) == pytest.approx(0.0156, abs=2e-4)  # 1.6 g per 100 g water
    assert max(ratios) == pytest.approx(0.0203, abs=2e-4)  # 2.0 g per 100 g water


def test_caffeine_stored_calibration_matches_a_recomputation(caffeine):
    stored = json.loads((CAFFEINE / calibrate_band.RESULT_NAME).read_text(encoding="utf-8"))

    def compare(left, right, path):
        if isinstance(left, dict):
            assert list(left) == list(right), path
            for key in left:
                compare(left[key], right[key], path + (key,))
        elif isinstance(left, str):
            assert left == right, path
        else:
            assert left == pytest.approx(right, rel=1e-5), path

    compare(stored, caffeine, ())


def test_caffeine_series_gives_the_load_while_dissolved_and_less_after_crystallization(caffeine):
    run_id = "low_seed_fast_cool"
    log_end = 19965.0  # [s] the temperature log of this run ends before its spectra do (22,982 s)

    for treatment in calibrate_band.TREATMENTS:
        time, ratio = calibrate_band.solute_ratio(CAFFEINE, run_id, treatment, caffeine)
        assert time[0] > 0.0 and time[-1] <= log_end
        hot = (time >= 5400.0) & (time <= 6500.0)
        assert np.median(ratio[hot]) == pytest.approx(caffeine["charge_ratio"], rel=5e-3)
        # The stored value converts the median height at the mean temperature; the
        # median of the converted series differs from it within the band scatter.
        assert np.median(ratio[time > log_end - 1800.0]) == pytest.approx(
            caffeine["runs"][run_id][treatment]["slurry_end_ratio"], rel=1e-3)
        assert ratio[-1] < 0.45 * caffeine["charge_ratio"]
    with pytest.raises(ValueError, match="Run 'high_seed_fast_cool' has no band calibration"):
        calibrate_band.solute_ratio(CAFFEINE, "high_seed_fast_cool", "common_baseline", caffeine)
