"""Caffeine checkpoint observations and native prediction at measured IR times."""

from pathlib import Path

import numpy as np
import pytest

from examples.pat import caffeine_m6


@pytest.mark.unit
def test_caffeine_case_runs_have_distinct_grids_and_preserve_measured_bands():
    runs = caffeine_m6.load_runs(caffeine_m6.CASE, "common_slope")

    assert tuple(runs) == (
        "high_seed_slow_cool", "low_seed_fast_cool", "low_seed_slow_cool"
    )
    # Inspection of the reduced source tables: the window starts after the
    # FBRM addition spike and ends at the first logged reading <= 20 C.
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


@pytest.mark.assimulo
@pytest.mark.integration
def test_native_caffeine_predictions_use_ir_band_per_solute_solvent_ratio():
    pytest.importorskip("assimulo")
    runs = caffeine_m6.load_runs(caffeine_m6.CASE, "common_slope")
    name = "high_seed_slow_cool"
    run = runs[name]
    # Follow the first, middle and last measured spectra from the fitted window.
    indices = [0, run.observation.x.size // 2, -1]
    time = run.observation.x[indices]  # [s] after seeding
    params = np.array([run.kinetics["growth"][0], run.kinetics["growth"][2]])  # [um/s], [-]

    band = caffeine_m6.predict_band(params, time, name, runs=runs)  # [-] absorbance
    # Compare with the inventory-basis relation using the same native solve.
    # This independently reconstructs the absorbance from vapor-free liquid
    # masses and the documented affine band model, including its T term.
    slurry = caffeine_m6.batch_cooling.simulate(
        run.cooling, run.kinetics, time, "legacy", solver_options=caffeine_m6.CVODE_OPTIONS
    )
    ratio = slurry.solute_mass / slurry.solvent_mass  # [kg caffeine/kg water]
    temperature = run.cooling.temperature(time)  # [K]
    calibration = run.calibration
    expected = calibration["baseline"] + (
        calibration["slope"]
        + calibration["slope_per_kelvin"] * (temperature - calibration["reference_temperature"])
    ) * ratio  # [-] absorbance
    # Both paths solve the same deterministic trajectory. This tolerance covers
    # the multiplication/addition but rejects composition and time-grid errors.
    np.testing.assert_allclose(band, expected, rtol=4 * np.finfo(float).eps, atol=0)
    assert np.all(np.diff(band) < 0)
