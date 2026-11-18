"""Seeded batch cooling simulation of the caffeine case: its inputs (no solver).

The kinetic constants of the description are starting values, not results. The
solver-backed checks are in ``test_pat_batch_cooling_models.py``.
"""
from pathlib import Path

import numpy as np
import pytest

from examples.pat import batch_cooling, dataset
from PharmaPy.Phases import LiquidPhase

pytestmark = pytest.mark.unit

CASE = Path(batch_cooling.__file__).parent / "caffeine"
RUN = "high_seed_slow_cool"


def test_case_inputs_come_from_the_description_and_the_solubility_curve():
    inputs, kinetics = batch_cooling.load_case(CASE, RUN, "common_slope")

    assert (inputs.solute, inputs.solvent) == ("caffeine", "water")
    assert (inputs.solute_mass, inputs.solvent_mass, inputs.seed_mass) == pytest.approx((0.0165434, 0.35, 0.5713e-3))
    assert inputs.seed_size == pytest.approx(31.6e-6)
    assert inputs.shape_factor == pytest.approx(np.pi / 6.0, rel=1e-12)
    assert inputs.solute_density == 1324.4
    assert Path(inputs.database).is_file()
    assert kinetics == {"growth": (0.05, 0.0, 1.0), "secondary_nucleation": (1e10, 0.0, 2.0, 1.0)}
    # Model time zero is the seeding spike, during the hold at 37 C; 600 s earlier
    # the liquid was still settling from the cooling step, and long after the run
    # it is 20 C.
    assert float(inputs.temperature(0.0)) - 273.15 == pytest.approx(36.96, abs=0.02)
    assert float(inputs.temperature(-600.0)) - 273.15 == pytest.approx(37.27, abs=0.05)
    assert float(inputs.temperature(30000.0)) - 273.15 == pytest.approx(20.0, abs=0.05)
    # Pooled curve of solubility.json: 1.85 g per 100 g water at 20 C, 4.51 at 40 C.
    assert 100.0 * inputs.saturated_ratio(np.array([293.15, 313.15])) == pytest.approx([1.85, 4.51], abs=0.01)
    other, _ = batch_cooling.load_case(CASE, "low_seed_fast_cool", "common_baseline")
    assert other.seed_mass == pytest.approx(0.1143e-3)
    assert 100.0 * float(other.saturated_ratio(293.15)) == pytest.approx(1.74, abs=0.01)


def test_seeding_time_of_each_run_is_its_fbrm_spike_in_the_seeding_hold():
    hold_start, hold_end = 7560.0, 9360.0  # [s] the 30 min hold at 37 C of the temperature program
    for run in dataset.load_description(CASE)["runs"]:
        fbrm = dataset.read_reduced(CASE, run["id"], "fbrm")
        counts = fbrm["counts_1_to_10um"] + fbrm["counts_10_to_100um"] + fbrm["counts_100_to_1000um"]
        hold = (fbrm["time"] >= hold_start) & (fbrm["time"] <= hold_end)
        spike = fbrm["time"][hold][np.argmax(counts[hold])]  # [s]
        assert run["seeding_time_s"] == pytest.approx(spike, abs=1.0), run["id"]
        assert counts[hold].max() > 300.0  # the clear solution reads below 60 counts


def test_measurement_times_are_the_samples_after_seeding_inside_the_temperature_log():
    spectra = batch_cooling.measurement_times(CASE, RUN, "ir")
    chords = batch_cooling.measurement_times(CASE, RUN, "fbrm")

    assert (spectra.size, chords.size) == (514, 3422)
    assert spectra[0] == pytest.approx(21.1, abs=0.1) and chords[0] == pytest.approx(0.2, abs=0.1)
    assert np.all(np.diff(spectra) > 0.0) and np.all(np.diff(chords) > 0.0)
    log_end = dataset.read_reduced(CASE, RUN, "temperature")["time"][-1] - 8762.0  # [s] model time
    assert spectra[-1] <= log_end and chords[-1] <= log_end
    assert batch_cooling.measurement_times(CASE, RUN, "ir", stop=1000.0)[-1] <= 1000.0


def test_property_file_gives_the_density_of_the_charge_by_ideal_mixing():
    # 16.5434 g caffeine (1344.3 kg/m3 dissolved) in 350 g water (996.6 kg/m3):
    # 0.3665434 kg in 1.2306e-5 + 3.51194e-4 m3, so 1008.4 kg/m3.
    liquid = LiquidPhase(str(CASE / "compounds.json"), temp=310.0, mass=0.3665434,
                         mass_frac=[0.0165434 / 0.3665434, 0.35 / 0.3665434])

    assert liquid.getDensity() == pytest.approx(1008.4, abs=0.1)
    assert liquid.vol == pytest.approx(3.6350e-4, rel=1e-4)


def test_rows_of_requested_times_are_found_and_a_missing_time_is_reported():
    reported = np.array([0.0, 10.0, 25.0, 40.0, 70.0])  # [s]

    np.testing.assert_array_equal(batch_cooling._rows_of(reported, np.array([10.0, 40.0, 70.0])), [1, 3, 4])
    with pytest.raises(RuntimeError, match="did not return 2 of 3 requested times, first 30 s"):
        batch_cooling._rows_of(reported, np.array([25.0, 30.0, 80.0]))


def test_inputs_that_cannot_be_simulated_are_refused():
    inputs, kinetics = batch_cooling.load_case(CASE, RUN, "common_slope")

    with pytest.raises(ValueError, match="Unknown run 'run_5'"):
        batch_cooling.load_case(CASE, "run_5", "common_slope")
    with pytest.raises(ValueError, match="Unknown treatment 'per_run'"):
        batch_cooling.load_case(CASE, RUN, "per_run")
    with pytest.raises(ValueError, match="Unknown model 'fvm'"):
        batch_cooling.simulate(inputs, kinetics, [10.0, 20.0], "fvm")
    for times in ([0.0, 10.0], [10.0, 10.0], [20.0, 10.0], []):
        with pytest.raises(ValueError, match="positive and strictly increasing"):
            batch_cooling.simulate(inputs, kinetics, times, "legacy")
