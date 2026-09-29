"""Regression tests for plotting unit metadata labels."""

import matplotlib.pyplot as plt

from PharmaPy.Plotting import format_unit_label, latexify_name, name_yaxes


def test_format_unit_label_accepts_bracketed_metadata():
    """Bracketed source metadata renders without nested brackets."""
    temp_units = "[K]"
    density_units = "[kg/m**3]"

    assert format_unit_label(temp_units) == latexify_name(temp_units[1:-1], units=True)
    assert format_unit_label(density_units) == latexify_name(
        density_units[1:-1], units=True)


def test_name_yaxes_suppresses_dimensionless_metadata():
    """Dimensionless ``[-]`` metadata does not add a plot-label suffix."""
    temp_units = "[K]"
    fig, axes = plt.subplots(1, 2)
    states = {
        "temp": {"units": temp_units},
        "x_liq": {"units": "[-]"},
    }

    try:
        name_yaxes(axes, states, ("temp", "x_liq"), ("T", "x_liq"), True)

        assert axes[0].get_ylabel() == (
            f"{latexify_name('T')} ({latexify_name(temp_units[1:-1], units=True)})"
        )
        assert axes[1].get_ylabel() == latexify_name("x_liq")
    finally:
        plt.close(fig)
