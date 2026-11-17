"""Solubility curve: the van't Hoff relation, the dataset workflow, and the caffeine runs.

The workflow is exercised on a small dataset written by the test for another
compound, generated from a known curve. The caffeine expectations were computed
from the reduced tables with a separate least-squares script.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from examples.pat import calibrate_band, fit_solubility, solubility, table_layout

pytestmark = pytest.mark.unit

CAFFEINE = Path(fit_solubility.__file__).parent / "caffeine"
INTERCEPT, SLOPE = 10.0, -4000.0  # [-], [K]: ln(ratio) is -3.3333 at 300 K and -2.5 at 320 K


def test_curve_and_saturation_temperature_are_inverse_to_each_other():
    ratios = solubility.saturated_ratio(np.array([300.0, 320.0]), INTERCEPT, SLOPE)

    np.testing.assert_allclose(ratios, [0.0356740, 0.0820850], rtol=1e-6)
    np.testing.assert_allclose(solubility.saturation_temperature(ratios, INTERCEPT, SLOPE), [300.0, 320.0])
    assert solubility.dissolution_enthalpy(SLOPE) == pytest.approx(33257.85, abs=0.01)  # 8.314462618 * 4000


def test_fit_recovers_the_constants_and_reports_the_scatter_about_the_curve():
    temperatures = np.array([290.0, 300.0, 325.0])  # [K]
    exact = solubility.saturated_ratio(temperatures, INTERCEPT, SLOPE)

    intercept, slope, scatter = solubility.fit_vant_hoff(temperatures, exact)
    assert (intercept, slope) == pytest.approx((INTERCEPT, SLOPE))
    assert scatter == pytest.approx(0.0, abs=1e-12)

    # Two points 10 % above and below the curve at each of two temperatures: the
    # fit keeps the curve and every residual of ln(ratio) is 0.1.
    doubled = np.repeat([290.0, 325.0], 2)
    spread = solubility.saturated_ratio(doubled, INTERCEPT, SLOPE) * np.exp([0.1, -0.1, 0.1, -0.1])
    intercept, slope, scatter = solubility.fit_vant_hoff(doubled, spread)
    assert (intercept, slope, scatter) == pytest.approx((INTERCEPT, SLOPE, 0.1))


def test_solubility_refuses_inputs_it_cannot_use():
    with pytest.raises(ValueError, match="two distinct temperatures"):
        solubility.fit_vant_hoff([300.0, 300.0], [0.03, 0.04])
    with pytest.raises(ValueError, match="must be positive"):
        solubility.fit_vant_hoff([300.0, 310.0], [0.03, 0.0])
    with pytest.raises(ValueError, match="equal one-dimensional shapes"):
        solubility.fit_vant_hoff([300.0, 310.0], [0.03])
    with pytest.raises(ValueError, match="must be positive"):
        solubility.saturation_temperature(0.0, INTERCEPT, SLOPE)
    with pytest.raises(ValueError, match="does not reach a ratio of 30000"):  # above exp(10) = 22026
        solubility.saturation_temperature(30000.0, INTERCEPT, SLOPE)


# A dataset of another compound that follows the curve above. The band reads
# 0.02 + 3.0 * ratio at every temperature. 10 g of solute in 100 g of solvent
# saturate at 325.14 K. The slurry is heated from 290 to 325 K between 1000 and
# 2000 s with solid present, is clear at 340 and 330 K, and is cooled back to
# 290 K, where the last spectra read a ratio of 0.03, above the curve. Run "run_b"
# has the same slope on a band baseline 0.005 higher.
RAMP_TIMES = np.arange(1000.0, 2001.0, 100.0)  # [s]
SATURATED_TIMES = np.concatenate(([100.0, 300.0, 500.0, 700.0], RAMP_TIMES))  # [s]
SATURATED_TEMPERATURES = np.concatenate(([290.0] * 4, 290.0 + 35.0 * (RAMP_TIMES - 1000.0) / 1000.0))  # [K]
CLEAR_TIMES = [2300.0, 2500.0, 2700.0, 3300.0, 3500.0, 3700.0]  # [s]
END_TIMES = [4300.0, 4500.0, 4700.0]  # [s]


@pytest.fixture()
def other_dataset(tmp_path):
    log_time = np.arange(0.0, 4801.0, 100.0)  # [s]
    log_temperature = np.interp(log_time, [0.0, 1000.0, 2000.0, 2200.0, 3000.0, 3200.0, 4000.0, 4200.0, 4800.0],
                                [290.0, 290.0, 325.0, 340.0, 340.0, 330.0, 330.0, 290.0, 290.0])  # [K]
    solid_present = (log_time <= 2000.0) | (log_time >= 4200.0)
    ratios = np.concatenate((solubility.saturated_ratio(SATURATED_TEMPERATURES, INTERCEPT, SLOPE), [0.1] * 6, [0.03] * 3))
    times = np.concatenate((SATURATED_TIMES, CLEAR_TIMES, END_TIMES))
    (tmp_path / "reduced").mkdir()
    for kind, columns in (("uv", {"peak": "-"}), ("thermo", {"liquid_temperature": "K"}),
                          ("particles", {"fine": "counts", "coarse": "counts"})):
        (tmp_path / ("reduced_%s.layout.json" % kind)).write_text(json.dumps({
            "header_row": 0, "units_row": 1, "first_data_row": 2,
            "columns": {"time": {"header": "time", "unit": "s", "role": "time"},
                        **{name: {"header": name, "unit": unit} for name, unit in columns.items()}}}))
    for run_id, offset, solvent_times in (("run_a", 0.0, [-300.0, -200.0, -100.0]), ("run_b", 0.005, [])):
        table_layout.write_table(
            tmp_path / "reduced" / ("%s_uv.csv" % run_id),
            {"time": np.concatenate((solvent_times, times)),
             "peak": np.concatenate(([0.02] * len(solvent_times), 0.02 + offset + 3.0 * ratios))},
            {"time": "s", "peak": "-"}, {"time": 1, "peak": 7})
        table_layout.write_table(tmp_path / "reduced" / ("%s_thermo.csv" % run_id),
                                 {"time": log_time, "liquid_temperature": log_temperature},
                                 {"time": "s", "liquid_temperature": "K"}, {"time": 1, "liquid_temperature": 3})
        table_layout.write_table(tmp_path / "reduced" / ("%s_particles.csv" % run_id),
                                 {"time": log_time, "fine": 300.0 * solid_present, "coarse": 200.0 * solid_present},
                                 {"time": "s", "fine": "counts", "coarse": "counts"}, {"time": 1, "fine": 1, "coarse": 1})
    (tmp_path / "runs.json").write_text(json.dumps({
        "charge": {"solute": "solute_a", "solvent": "ethanol", "solute_mass_g": 10.0, "solvent_mass_g": 100.0},
        "band_calibration": {
            "table": "uv", "height_column": "peak", "temperature_table": "thermo",
            "temperature_column": "liquid_temperature", "runs": ["run_a", "run_b"], "reference_run": "run_a",
            "reference_temperature_K": 300.0, "solvent_window_s": [-350.0, -50.0],
            "dissolved_windows_s": [[2200.0, 2800.0], [3200.0, 3800.0]], "slurry_start_window_s": [50.0, 750.0],
            "slurry_end_duration_s": 500.0},
        "solubility": {"window_s": [50.0, 2600.0], "solid_table": "particles", "solid_columns": ["fine", "coarse"],
                       "minimum_solid_signal": 100.0}}))
    calibrate_band.main([str(tmp_path)])
    return tmp_path


def test_workflow_recovers_the_curve_of_a_dataset_of_another_compound(other_dataset):
    result = fit_solubility.fit(other_dataset)

    assert (result["solute"], result["solvent"], result["charge_ratio"]) == ("solute_a", "ethanol", 0.1)
    exact = [result["treatments"][treatment]["runs"]["run_a"] for treatment in calibrate_band.TREATMENTS]
    exact += [result["treatments"]["common_slope"][name] for name in ("pooled",)]
    exact.append(result["treatments"]["common_slope"]["runs"]["run_b"])
    for curve in exact:
        # Only the fifteen spectra with solid inside the window: not the clear
        # solution at 340 K, and not the late spectra that read above the curve.
        assert curve["count"] in (15, 30)
        assert (curve["intercept"], curve["slope_K"]) == pytest.approx((INTERCEPT, SLOPE), rel=1e-3)
        assert curve["scatter"] < 1e-4
        assert (curve["minimum_temperature_K"], curve["maximum_temperature_K"]) == (290.0, 325.0)
        assert curve["charge_saturation_temperature_K"] == pytest.approx(325.14, abs=0.02)  # 4000 / (10 - ln 0.1)
    # Read on the baseline of the other run with the slope 3.05 that its plateaus
    # then give, run_b reads (0.005 + 3 * ratio) / 3.05: right at the charge and
    # 0.0238 instead of 0.0225 at 290 K. The runs then disagree, which the pooled
    # scatter shows.
    shifted = result["treatments"]["common_baseline"]
    run_b = shifted["runs"]["run_b"]
    assert float(solubility.saturated_ratio(290.0, run_b["intercept"], run_b["slope_K"])) == pytest.approx(0.0238, rel=0.01)
    assert shifted["pooled"]["scatter"] > 100 * result["treatments"]["common_slope"]["pooled"]["scatter"]


def test_workflow_writes_the_result_file(other_dataset):
    fit_solubility.main([str(other_dataset)])

    stored = json.loads((other_dataset / fit_solubility.RESULT_NAME).read_text(encoding="utf-8"))
    assert stored["treatments"]["common_slope"]["pooled"]["slope_K"] == pytest.approx(SLOPE, rel=1e-3)
    assert list(stored["treatments"]) == list(calibrate_band.TREATMENTS)


@pytest.fixture(scope="module")
def caffeine():
    return fit_solubility.fit(CAFFEINE)


@pytest.mark.parametrize("treatment, scatter, at_20C, at_40C, saturation_C", [
    ("common_baseline", 0.078, 1.74, 4.52, 41.0),
    ("common_slope", 0.045, 1.85, 4.51, 41.1),
])
def test_caffeine_pooled_curve_under_each_treatment(caffeine, treatment, scatter, at_20C, at_40C, saturation_C):
    curve = caffeine["treatments"][treatment]["pooled"]

    assert curve["count"] == 177
    assert curve["scatter"] == pytest.approx(scatter, abs=0.002)
    grams_per_100_g = 100.0 * solubility.saturated_ratio(np.array([293.15, 313.15]), curve["intercept"], curve["slope_K"])
    assert grams_per_100_g == pytest.approx([at_20C, at_40C], abs=0.01)
    assert curve["charge_saturation_temperature_K"] - 273.15 == pytest.approx(saturation_C, abs=0.1)
    assert curve["minimum_temperature_K"] - 273.15 == pytest.approx(19.6, abs=0.1)
    assert 39.0 < curve["maximum_temperature_K"] - 273.15 < 41.5  # solid is gone by the clear point


def test_caffeine_runs_agree_better_when_they_share_the_slope(caffeine):
    def at_20C(treatment):
        return [100.0 * float(solubility.saturated_ratio(293.15, curve["intercept"], curve["slope_K"]))
                for curve in caffeine["treatments"][treatment]["runs"].values()]  # [g per 100 g water]

    assert at_20C("common_baseline") == pytest.approx([1.95, 1.61, 1.63], abs=0.01)
    assert at_20C("common_slope") == pytest.approx([1.95, 1.77, 1.80], abs=0.01)


def test_caffeine_stored_curve_matches_a_recomputation(caffeine):
    stored = json.loads((CAFFEINE / fit_solubility.RESULT_NAME).read_text(encoding="utf-8"))

    for treatment in calibrate_band.TREATMENTS:
        for name, curve in [("pooled", caffeine["treatments"][treatment]["pooled"])] + list(
                caffeine["treatments"][treatment]["runs"].items()):
            kept = stored["treatments"][treatment]["pooled"] if name == "pooled" else stored["treatments"][treatment]["runs"][name]
            assert list(kept) == list(curve)
            assert list(kept.values()) == pytest.approx(list(curve.values()), rel=1e-5), (treatment, name)
