"""Reduction of runs to the whole-spectrum IR score, and the committed azithromycin tables.

The steps run on constructed spectra with known composition and on the excerpts in
``tests/fixtures/pat``. The raw azithromycin runs are not in the repository; their
committed tables are checked against the run design in ``runs.json``.
"""
import copy
import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from examples.pat import block_package, dataset, ic_spectra, reduce_runs

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "pat"
CASE = Path(dataset.__file__).parent / "azithromycin"
DESCRIPTION = dataset.load_description(CASE)
RUNS = {run["id"]: run for run in DESCRIPTION["runs"]}
ZERO_CELSIUS = 273.15  # [K]
FINAL_RAMP_START = 9360.0  # [s] end of the 37 C hold in the setpoint program
ROUND_OFF = 1e-9  # [-] the decomposition is exact for constructed spectra

AXIS = np.linspace(1800.0, 900.0, 46)  # [cm-1]


def _band(center: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((AXIS - center) / width) ** 2)


SOLVENT = _band(1740.0, 20.0, 1.7) + _band(1240.0, 25.0, 2.0) + _band(1045.0, 18.0, 1.6) + 0.05  # [-]
SOLUTE = _band(1000.0, 30.0, 0.35) + _band(1170.0, 15.0, 0.30) + _band(1460.0, 22.0, 0.12)  # [-] per [kg/kg]
PER_KELVIN = 1e-3 * (_band(1700.0, 40.0, -1.0) + _band(1100.0, 60.0, 0.6)) + 2e-5  # [1/K]
REFERENCE_TEMPERATURE = 310.0  # [K]
SETTINGS = {"reference_run": "second", "comparison_run": "first", "reference_temperature_K": REFERENCE_TEMPERATURE,
            "clear_solution_window_s": [300.0, 600.0], "dissolved_window_s": [200.0, 260.0], "slurry_window_s": [0.0, 60.0],
            "wavenumber_window_cm-1": [950.0, 1700.0], "maximum_absorbance": 1.3}


def _spectra(ratio, temperature, scale=0.0):
    """Constructed spectra of a solution with a dissolved ratio [kg/kg] at a temperature [K]."""
    ratio, temperature, scale = np.broadcast_arrays(np.asarray(ratio, float), np.asarray(temperature, float), np.asarray(scale, float))
    clean = SOLVENT + np.outer(ratio, SOLUTE) + np.outer(temperature - REFERENCE_TEMPERATURE, PER_KELVIN)
    return clean * (1.0 + scale[:, None])


def _constructed_runs():
    # Second run, the reference: slurry at 0.20, its charge 0.45 dissolved, then the same charge while cooling.
    time_2 = np.array([0.0, 30.0, 60.0, 200.0, 230.0, 260.0, 300.0, 400.0, 500.0, 600.0, 700.0])  # [s]
    ratio_2 = np.array([0.20, 0.20, 0.20, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.35])  # [kg/kg]
    temperature_2 = np.array([293.0, 293.0, 293.1, 310.2, 310.1, 310.0, 309.0, 304.0, 299.0, 294.0, 293.0])  # [K]
    # First run: a charge of 0.40, then 0.30, the last spectrum with 2 % more signal from a change of probe contact.
    time_1 = np.array([200.0, 240.0, 500.0, 520.0])  # [s]
    ratio_1 = np.array([0.40, 0.40, 0.30, 0.30])  # [kg/kg]
    temperature_1 = np.array([310.0, 310.3, 300.0, REFERENCE_TEMPERATURE])  # [K]
    scale_1 = np.array([0.0, 0.0, 0.0, 0.02])  # [-]
    return {"first": (time_1, temperature_1, AXIS, _spectra(ratio_1, temperature_1, scale_1)),
            "second": (time_2, temperature_2, AXIS, _spectra(ratio_2, temperature_2))}


def test_score_tables_count_the_dissolved_solute_in_units_of_the_reference_drop():
    tables, summary = reduce_runs.score_tables(_constructed_runs(), {"ir_score": SETTINGS})

    columns, units = tables["second"]
    assert list(columns) == ["time", "score", "scale", "offset", "tilt", "residual"]
    assert units == {"time": "s", "score": "-", "scale": "-", "offset": "-", "tilt": "-", "residual": "-"}
    # One unit is the drop from the dissolved charge to the slurry of the reference run, 0.45 - 0.20 kg/kg.
    np.testing.assert_allclose(columns["score"], (np.array([0.20] * 3 + [0.45] * 7 + [0.35]) - 0.45) / 0.25, atol=ROUND_OFF)
    np.testing.assert_allclose(columns["residual"], 0.0, atol=ROUND_OFF)
    # The first run counts from its own charge, 0.40 kg/kg: 0.30 reads (0.30 - 0.40) / 0.25. The gain of the last
    # spectrum goes to the scale term; it multiplies the solute change as well, so the score reads 2 % larger.
    first = tables["first"][0]
    np.testing.assert_allclose(first["score"], [0.0, 0.0, -0.4, -0.4 * 1.02], atol=ROUND_OFF)
    np.testing.assert_allclose(first["scale"], [0.0, 0.0, 0.0, 0.02], atol=ROUND_OFF)
    assert first["time"].shape == (4,) and columns["time"].shape == (11,)
    against = summary["dissolved_spectrum_against_comparison_run"]
    assert against["first"]["score"] == pytest.approx(0.0, abs=ROUND_OFF)
    assert against["second"]["score"] == pytest.approx((0.45 - 0.40) / 0.25, abs=ROUND_OFF)
    assert (against["first"]["dissolved_spectra"], against["second"]["dissolved_spectra"]) == (2, 3)
    assert (summary["clear_solution_spectra"], summary["slurry_spectra"]) == (4, 3)
    assert (summary["clear_solution_temperature_min_K"], summary["clear_solution_temperature_max_K"]) == (294.0, 309.0)
    # 38 grid points lie between 950 and 1700 cm-1; four are on solvent bands above 1.3.
    assert summary["spectral_points"] == 34
    assert summary["temperature_effect_residual"] < ROUND_OFF


def test_score_tables_refuse_an_empty_window_and_a_different_axis():
    runs = _constructed_runs()
    settings = dict(SETTINGS, dissolved_window_s=[200.0, 230.0])
    time, temperature, axis, spectra = runs["first"]
    late = {"first": (time + 100.0, temperature, axis, spectra), "second": runs["second"]}
    with pytest.raises(ValueError, match="Run first has no spectrum in the dissolved window"):
        reduce_runs.score_tables(late, {"ir_score": settings})
    shifted = {"first": (time, temperature, axis + 1.0, spectra), "second": runs["second"]}
    with pytest.raises(ValueError, match="Run first does not share the spectral axis"):
        reduce_runs.score_tables(shifted, {"ir_score": SETTINGS})


def test_spectra_outside_the_temperature_log_are_left_out():
    time = np.array([-30.0, 10.0, 45.0, 100.0, 130.0])  # [s]
    spectra = np.arange(10.0).reshape(5, 2)
    log_time, log_temperature = np.array([0.0, 60.0, 120.0]), np.array([293.0, 295.0, 301.0])  # [s], [K]

    kept_time, temperature, axis, kept = reduce_runs.spectra_with_temperature(time, AXIS[:2], spectra, log_time, log_temperature)

    np.testing.assert_allclose(kept_time, [10.0, 45.0, 100.0])
    np.testing.assert_allclose(temperature, [293.0 + 2.0 * 10.0 / 60.0, 293.0 + 2.0 * 45.0 / 60.0, 295.0 + 6.0 * 40.0 / 60.0])
    np.testing.assert_array_equal(kept, spectra[1:4])
    np.testing.assert_array_equal(axis, AXIS[:2])


def test_run_clock_can_be_given_as_the_local_start_of_the_log():
    run = {"id": "r", "temperature_log_start_local": "2026-10-21T16:14:37"}

    assert reduce_runs.run_clock_origin(None, run, DESCRIPTION) == datetime.datetime(2026, 10, 21, 16, 14, 37)
    both = dict(run, temperature_log_start_after_fbrm_start_s=20.0)
    with pytest.raises(ValueError, match="exactly one of"):
        reduce_runs.run_clock_origin(datetime.datetime(2026, 10, 21, 16, 14), both, DESCRIPTION)
    with pytest.raises(ValueError, match="exactly one of"):
        reduce_runs.run_clock_origin(datetime.datetime(2026, 10, 21, 16, 14), {"id": "r"}, DESCRIPTION)
    with pytest.raises(ValueError, match="has no FBRM samples"):
        reduce_runs.run_clock_origin(None, {"id": "r", "temperature_log_start_after_fbrm_start_s": 20.0}, DESCRIPTION)


def test_declared_spectra_without_metadata_are_left_out():
    infos = ic_spectra.spectrum_infos((FIXTURES / "spectrum_info_tags.xml").read_text(encoding="utf-8"))
    stream = block_package.read_package((FIXTURES / "ir_head_blocks.iCIR").read_bytes())["/ReactionSpectras.vector"][1]
    records = dict(ic_spectra.spectrum_records(stream))
    orphans = sum(identifier not in infos for identifier in records)
    origin = datetime.datetime(2026, 10, 24, 0, 49, 17)  # local time; the spectra carry UTC stamps four hours ahead

    time, axis, spectra = reduce_runs.timed_spectra(infos, records, origin, -4.0, without_metadata=orphans)

    assert orphans > 0
    # The three spectra with metadata were taken at 04:29:10.257475, 04:30:11.952319 and 04:31:10.578463 UTC.
    np.testing.assert_allclose(time, [-1206.742525, -1145.047681, -1086.421537], atol=1e-6)
    assert spectra.shape == (3, axis.size) and axis[0] > axis[-1]
    with pytest.raises(ValueError, match="%d spectra without metadata were declared, but %d were found" % (orphans + 1, orphans)):
        reduce_runs.timed_spectra(infos, records, origin, -4.0, without_metadata=orphans + 1)


def test_a_description_needs_exactly_one_ir_block_and_a_boolean_fbrm_flag(tmp_path):
    run = dict(RUNS["high_seed_slow_cool"])
    both = copy.deepcopy(DESCRIPTION)
    both["ir_band"] = {"peak_cm-1": 1000.0}
    neither = {key: value for key, value in DESCRIPTION.items() if key != "ir_score"}
    for description in (both, neither):
        with pytest.raises(ValueError, match="exactly one of the blocks"):
            reduce_runs.reduce_run(CASE, tmp_path, run, description, tmp_path / "out")
    with pytest.raises(ValueError, match="has_fbrm_samples of run high_seed_slow_cool must be true or false"):
        reduce_runs.reduce_run(CASE, tmp_path, dict(run, has_fbrm_samples="no"), DESCRIPTION, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("run_id", sorted(RUNS))
def test_reduced_azithromycin_tables_follow_the_run_design(run_id):
    run = RUNS[run_id]
    temperature = dataset.read_reduced(CASE, run_id, "temperature")
    ir = dataset.read_reduced(CASE, run_id, "ir")

    # Program: 45 C at the top, 37 C at seeding, then a ramp at the run's rate.
    assert temperature["external_temperature"].max() == pytest.approx(45.0 + ZERO_CELSIUS, abs=0.1)
    in_hold = (temperature["time"] > 7800.0) & (temperature["time"] < FINAL_RAMP_START)
    assert np.median(temperature["setpoint_temperature"][in_hold]) == pytest.approx(DESCRIPTION["seeding"]["temperature_C"] + ZERO_CELSIUS, abs=0.01)
    on_ramp = (temperature["time"] > FINAL_RAMP_START + 300.0) & (temperature["time"] < FINAL_RAMP_START + 1500.0)
    slope = np.polyfit(temperature["time"][on_ramp], temperature["setpoint_temperature"][on_ramp], 1)[0]  # [K/s]
    assert -slope * 60.0 == pytest.approx(run["final_ramp_rate_C_per_min"], rel=0.02)
    # Every spectrum in the table was taken while the temperature was logged.
    assert temperature["time"][0] <= ir["time"][0] and ir["time"][-1] <= temperature["time"][-1]
    # The score counts from the run's own dissolved charge: zero in the 37 C hold before seeding.
    window = DESCRIPTION["ir_score"]["dissolved_window_s"]
    dissolved = (ir["time"] >= window[0]) & (ir["time"] <= window[1])
    assert dissolved.sum() >= 3 and np.mean(ir["score"][dissolved]) == pytest.approx(0.0, abs=1e-4)
    # The charge is dissolved in the 45 C hold too. After seeding the dissolved solute falls by at least three
    # quarters of the drop between the dissolved charge and the 20 C slurry of the reference run.
    hot = (ir["time"] > 5400.0) & (ir["time"] < 6500.0)
    assert abs(np.median(ir["score"][hot])) < 0.02
    assert ir["score"][ir["time"] > run["seeding_time_s"]].min() < -0.75
    assert ir["residual"].max() < 0.005
    has_fbrm = run.get("has_fbrm_samples", True)
    assert (CASE / "reduced" / (run_id + "_fbrm.csv")).exists() == has_fbrm
    if has_fbrm:
        fbrm = dataset.read_reduced(CASE, run_id, "fbrm")
        total = fbrm["counts_1_to_10um"] + fbrm["counts_10_to_100um"] + fbrm["counts_100_to_1000um"]  # [counts]
        assert np.median(np.diff(fbrm["time"])) == pytest.approx(DESCRIPTION["fbrm"]["sample_interval_s"], abs=0.5)
        # seeding_time_s is the sample that ends the largest rise between consecutive samples in the 37 C hold.
        hold = (fbrm["time"] > 7700.0) & (fbrm["time"] < FINAL_RAMP_START)
        rise_end = fbrm["time"][hold][1:][np.argmax(np.diff(total[hold]))]
        assert rise_end == pytest.approx(run["seeding_time_s"], abs=1.0)
        # A higher seed load gives more counts after seeding.
        after = (fbrm["time"] > run["seeding_time_s"] + 60.0) & (fbrm["time"] < run["seeding_time_s"] + 300.0)
        assert (np.median(total[after]) > 500.0) == (run["seed_percent_of_yield"] == 5.0)


def test_reference_slurry_reads_minus_one_and_the_summary_matches_the_tables():
    settings = DESCRIPTION["ir_score"]
    summary = json.loads((CASE / "reduced" / reduce_runs.SCORE_SUMMARY).read_text(encoding="utf-8"))
    ir = dataset.read_reduced(CASE, settings["reference_run"], "ir")
    slurry = (ir["time"] >= settings["slurry_window_s"][0]) & (ir["time"] <= settings["slurry_window_s"][1])

    assert np.mean(ir["score"][slurry]) == pytest.approx(-1.0, abs=1e-4)
    assert summary["slurry_spectra"] == slurry.sum()
    assert (summary["reference_run"], summary["comparison_run"]) == (settings["reference_run"], settings["comparison_run"])
    against = summary["dissolved_spectrum_against_comparison_run"]
    assert sorted(against) == sorted(RUNS) and against[settings["comparison_run"]]["score"] == 0.0
    for run_id, terms in against.items():
        ir = dataset.read_reduced(CASE, run_id, "ir")
        window = settings["dissolved_window_s"]
        assert terms["dissolved_spectra"] == ((ir["time"] >= window[0]) & (ir["time"] <= window[1])).sum()
