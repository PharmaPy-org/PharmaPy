"""Quality figures of the reduced caffeine runs: panels, units and the data they draw."""
import numpy as np
import pytest

from examples.pat.caffeine import plot_runs

pytestmark = pytest.mark.unit

RUN = "low_seed_fast_cool"  # has samples without counts, so the mean chord has gaps


def test_figure_draws_the_reduced_tables_in_five_labelled_panels(tmp_path):
    temperature = plot_runs.read_reduced(RUN, "temperature")
    fbrm = plot_runs.read_reduced(RUN, "fbrm")
    ir = plot_runs.read_reduced(RUN, "ir")

    figure = plot_runs.plot_run(RUN)
    panels = figure.axes

    assert [panel.get_ylabel() for panel in panels] == [
        "Temperature [C]", "FBRM counts [counts]", "Mean chord, square weighted [um]",
        "IR band height [-]", "IR solvent absorbance [-]"]
    assert panels[-1].get_xlabel() == "Time on the run clock [h]"
    assert [len(panel.get_lines()) for panel in panels] == [2, 3, 1, 1, 1]
    assert [text.get_text() for text in panels[0].get_legend().get_texts()] == ["external", "setpoint"]
    assert [text.get_text() for text in panels[1].get_legend().get_texts()] == ["1 to 10 um", "10 to 100 um", "100 to 1000 um"]
    external, setpoint = panels[0].get_lines()
    np.testing.assert_allclose(external.get_xdata(), temperature["time"] / 3600.0)
    np.testing.assert_allclose(external.get_ydata(), temperature["external_temperature"] - 273.15)
    np.testing.assert_allclose(setpoint.get_ydata(), temperature["setpoint_temperature"] - 273.15)
    np.testing.assert_allclose(panels[1].get_lines()[2].get_ydata(), np.maximum(fbrm["counts_100_to_1000um"], 1.0))
    assert panels[1].get_yscale() == "log"
    np.testing.assert_allclose(panels[3].get_lines()[0].get_ydata(), ir["band_height"])
    np.testing.assert_allclose(panels[4].get_lines()[0].get_ydata(), ir["solvent_absorbance"])
    figure.savefig(tmp_path / "run.png")
    assert (tmp_path / "run.png").stat().st_size > 10000


def test_mean_chord_is_drawn_only_for_samples_with_enough_counts():
    fbrm = plot_runs.read_reduced(RUN, "fbrm")
    total = fbrm["counts_1_to_10um"] + fbrm["counts_10_to_100um"] + fbrm["counts_100_to_1000um"]  # [counts]

    drawn = plot_runs.plot_run(RUN).axes[2].get_lines()[0].get_ydata()

    sparse = total < 100.0
    assert 500 < sparse.sum() < total.size  # the clear solution spans hundreds of samples
    assert np.all(np.isnan(drawn[sparse]))
    np.testing.assert_allclose(drawn[~sparse], fbrm["mean_chord_square_weighted"][~sparse])


def test_unknown_run_is_refused():
    with pytest.raises(ValueError, match="Unknown run 'run_5'"):
        plot_runs.plot_run("run_5")
