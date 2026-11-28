"""Caffeine checkpoint observations and native prediction at measured IR times."""

from pathlib import Path
from shutil import copytree

import numpy as np
import pytest

from examples.pat import caffeine_m6
from PharmaPy.ParamEstim import Experiment
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision, ParameterRecord, Transformation


@pytest.mark.unit
def test_caffeine_case_runs_have_distinct_grids_and_preserve_measured_bands():
    runs = caffeine_m6.load_runs(caffeine_m6.CASE, "common_slope")

    assert tuple(runs) == (
        "high_seed_slow_cool", "low_seed_fast_cool", "low_seed_slow_cool"
    )
    # Inspection of the reduced source tables: the window starts after the
    # seeding time inferred in runs.json and ends at the first logged reading <= 20 C.
    expected = {
        "high_seed_slow_cool": (176, 21.1, 11307.6),
        "low_seed_fast_cool": (46, 29.0, 3209.9),
        "low_seed_slow_cool": (9, 195.3, 7749.1),
    }  # (spectrum count, first and last times after seeding [s])
    for name, (count, first, last) in expected.items():
        measurement = runs[name].observation
        assert measurement.x.size == count
        assert [measurement.x[0], measurement.x[-1]] == pytest.approx([first, last])
        assert measurement.x_units == "s"
        assert measurement.units == "1"

        # Match every band height back to its original row on the run clock.
        ir_path = Path(caffeine_m6.CASE, "reduced", f"{name}_ir.csv")
        source = np.genfromtxt(
            ir_path, delimiter=",", skip_header=2,
            names=("time", "band_height", "solvent_absorbance"),
        )
        rows = np.isin(
            np.round(source["time"], decimals=1),
            np.round(measurement.x + runs[name].seeding_time, decimals=1),
        )  # compare on the published tenth-second clock [s]
        np.testing.assert_array_equal(measurement.values, source["band_height"][rows])
    assert not np.array_equal(
        runs["high_seed_slow_cool"].observation.x, runs["low_seed_fast_cool"].observation.x
    )


@pytest.mark.unit
def test_replay_rejects_a_changed_reduced_table_before_running_solver(tmp_path):
    case = copytree(caffeine_m6.CASE, tmp_path / "caffeine")
    description = caffeine_m6.dataset.load_description(case)
    measurements, model_files = caffeine_m6.case_inputs(case, description)
    runs = caffeine_m6.load_runs(case, "common_slope")
    revision = "source fixture"
    growth = runs[caffeine_m6.FIT_RUNS[0]].kinetics["growth"]  # [um/s], [J/mol], [-]
    record = CalibrationRecord(
        caffeine_m6.CASE_ID, {
            name: Experiment({"ir_band": runs[name].observation}, args=(name,))
            for name in caffeine_m6.FIT_RUNS
        },
        (ParameterRecord("k_g", "um/s", growth[0]), ParameterRecord("g", "1", growth[2])),
        [DataSource(name, "experimental", "Reduced data", sha256=digest)
         for name, digest in measurements.items()],
        ModelRevision(caffeine_m6.MODEL_NAME, revision=revision),
        transformations=[Transformation(
            "Use the measured bands in the named calibration treatment.",
            parameters={"source_sha256": model_files, "treatment": "common_slope"},
        )],
    )
    output = tmp_path / "caffeine.json"
    record.save(output)

    # Keep the table parseable while changing its recorded source bytes.
    filename = "reduced/high_seed_slow_cool_ir.csv"
    with (case / filename).open("a", encoding="utf-8") as source:
        source.write("\n")
    with pytest.raises(ValueError, match="high_seed_slow_cool_ir.csv: the bytes differ"):
        caffeine_m6.replay_caffeine(case, CalibrationRecord.load(output), revision)


@pytest.mark.unit
def test_legacy_caffeine_refuses_to_drop_continuous_reporting():
    run = caffeine_m6.load_runs(caffeine_m6.CASE, "common_slope")["high_seed_slow_cool"]
    with pytest.raises(ValueError, match="requires continuous reporting"):
        caffeine_m6.batch_cooling.simulate(
            run.cooling, run.kinetics, run.observation.x[:1], "legacy",
            solver_options={"report_continuously": False},
        )


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("growth_slot", [0, 2], ids=["prefactor", "order"])
def test_native_caffeine_predictions_use_both_growth_coordinates_and_ir_basis(growth_slot):
    pytest.importorskip("assimulo")
    runs = caffeine_m6.load_runs(caffeine_m6.CASE, "common_slope")
    name = "high_seed_slow_cool"
    run = runs[name]
    # Follow the first, middle and last measured spectra from the fitted window.
    indices = [0, run.observation.x.size // 2, -1]
    time = run.observation.x[indices]  # [s] after seeding
    growth = list(run.kinetics["growth"])  # [um/s], [J/mol], [-]
    # Doubling each free coordinate separately exposes a wiring error that
    # would otherwise be invisible at the initial parameter values.
    growth[growth_slot] *= 2
    params = np.array([growth[0], growth[2]])  # [um/s], [-]

    band = caffeine_m6.predict_band(params, time, name, runs=runs)  # [-] absorbance
    # Set the native simulator's perturbed rate directly and reconstruct
    # absorbance from its liquid masses and the documented band equation.
    kinetics = dict(run.kinetics, growth=tuple(growth))  # [um/s], [J/mol], [-] for growth
    slurry = caffeine_m6.batch_cooling.simulate(
        run.cooling, kinetics, time, "legacy", solver_options=caffeine_m6.CVODE_OPTIONS
    )
    ratio = slurry.solute_mass / slurry.solvent_mass  # [kg caffeine/kg water]
    temperature = run.cooling.temperature(time)  # [K]
    calibration = run.calibration
    response = (
        calibration["slope"]
        + calibration["slope_per_kelvin"] * (temperature - calibration["reference_temperature"])
    ) * ratio  # [-] absorbance
    expected = calibration["baseline"] + response  # [-] absorbance
    # Four roundoff allowances cover the multiplication/addition at this
    # absolute band scale. This tests the handoff and equation, with the same
    # simulator underlying both trajectories.
    roundoff = 4 * np.finfo(float).eps * (
        abs(calibration["baseline"]) + np.max(np.abs(response))
    )  # [-] absorbance
    np.testing.assert_allclose(band, expected, rtol=0, atol=roundoff)
    assert np.all(np.diff(band) < 0)


@pytest.mark.assimulo
@pytest.mark.integration
def test_refactored_caffeine_retains_requested_rows_with_solver_options():
    pytest.importorskip("assimulo")
    run = caffeine_m6.load_runs(caffeine_m6.CASE, "common_slope")["high_seed_slow_cool"]
    # The two endpoint spectra span the cooling ramp, with nonzero crystal growth.
    time = run.observation.x[[0, -1]]  # [s] after seeding
    slurry = caffeine_m6.batch_cooling.simulate(
        run.cooling, run.kinetics, time, "refactored", solver_options=caffeine_m6.CVODE_OPTIONS
    )
    total = slurry.solute_mass + caffeine_m6.SlurryMeasurements.solid_mass(
        slurry, run.cooling.solute_density, run.cooling.shape_factor
    )  # [kg] caffeine
    np.testing.assert_array_equal(slurry.time, time)
    np.testing.assert_allclose(
        total, run.cooling.solute_mass + run.cooling.seed_mass,
        rtol=caffeine_m6.BALANCE_RTOL, atol=0,
    )
    assert slurry.solute_mass[1] < slurry.solute_mass[0]
