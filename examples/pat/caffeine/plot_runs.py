"""Quality figures of the reduced caffeine runs.

Usage, from the repository root::

    python -m examples.pat.caffeine.plot_runs "<output folder>"

One figure per run with five stacked panels on the run clock: temperatures, FBRM
counts per band, mean chord, IR band height, and the IR solvent-band contact check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .. import dataset
from ..reduce_runs import MEAN_CHORD

CASE = Path(__file__).parent

SECONDS_PER_HOUR = 3600.0  # [s/h]
ZERO_CELSIUS = 273.15  # [K]
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # fixed order: first, second, third series of a panel
SURFACE, TEXT, MUTED_TEXT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FIGURE_SIZE = (9.0, 11.0)  # [inch]
COUNT_FLOOR = 1.0  # [counts] lower limit of the logarithmic count axis; below it is noise of an empty window
# Display choice, not a physical limit: with fewer counts than this in a sample the
# weighted mean is set by a handful of chords and scatters over the whole axis.
MIN_COUNTS_FOR_MEAN = 100.0  # [counts]


def read_reduced(run_id: str, kind: str) -> dict:
    """Read one reduced table of a run through its layout.

    Parameters
    ----------
    run_id : str
        Run identifier from ``runs.json``.
    kind : str
        ``"temperature"``, ``"fbrm"`` or ``"ir"``.

    Returns
    -------
    dict
        ``{column: values}``; times in [s], temperatures in [K], counts in
        [counts], mean chord in [um], IR columns dimensionless.
    """
    return dataset.read_reduced(CASE, run_id, kind)


def plot_run(run_id: str) -> Figure:
    """Draw the quality figure of one run.

    Parameters
    ----------
    run_id : str
        Run identifier from ``runs.json``.

    Returns
    -------
    matplotlib.figure.Figure
        Five panels sharing the time axis [h, run clock]: external and setpoint
        temperature [C], FBRM counts per band [counts] on a logarithmic axis, the
        square-weighted mean chord [um] where a sample has at least
        `MIN_COUNTS_FOR_MEAN` counts, the IR band height [-] and the IR absorbance
        at the solvent band [-].

    Raises
    ------
    ValueError
        If ``run_id`` is not in ``runs.json``.
    """
    description = dataset.load_description(CASE)
    if run_id not in {run["id"] for run in description["runs"]}:
        raise ValueError("Unknown run %r; runs.json lists %s." % (run_id, [run["id"] for run in description["runs"]]))
    temperature, fbrm, ir = (read_reduced(run_id, kind) for kind in ("temperature", "fbrm", "ir"))
    figure = Figure(figsize=FIGURE_SIZE, facecolor=SURFACE, layout="constrained")
    panels = figure.subplots(5, 1, sharex=True)
    hours = {name: table["time"] / SECONDS_PER_HOUR for name, table in (("temperature", temperature), ("fbrm", fbrm), ("ir", ir))}

    panels[0].plot(hours["temperature"], temperature["external_temperature"] - ZERO_CELSIUS, color=SERIES_COLORS[0], label="external")
    panels[0].plot(hours["temperature"], temperature["setpoint_temperature"] - ZERO_CELSIUS, color=SERIES_COLORS[1], label="setpoint")
    panels[0].set_ylabel("Temperature [C]")
    band_edges = description["fbrm"]["band_edges_um"]  # [um]
    total = 0.0  # [counts]
    for index, color in enumerate(SERIES_COLORS):
        name = "counts_%g_to_%gum" % (band_edges[index], band_edges[index + 1])
        total = total + fbrm[name]
        panels[1].plot(hours["fbrm"], fbrm[name].clip(COUNT_FLOOR), color=color, label="%g to %g um" % (band_edges[index], band_edges[index + 1]))
    panels[1].set_yscale("log")
    panels[1].set_ylabel("FBRM counts [counts]")
    panels[2].plot(hours["fbrm"], np.where(total >= MIN_COUNTS_FOR_MEAN, fbrm[MEAN_CHORD], np.nan), color=SERIES_COLORS[0])
    panels[2].set_ylabel("Mean chord, square weighted [um]")
    panels[3].plot(hours["ir"], ir["band_height"], color=SERIES_COLORS[0], marker="o", markersize=2.5)
    panels[3].set_ylabel("IR band height [-]")
    panels[4].plot(hours["ir"], ir["solvent_absorbance"], color=SERIES_COLORS[0], marker="o", markersize=2.5)
    panels[4].set_ylabel("IR solvent absorbance [-]")
    panels[4].set_ylim(0.0, 1.05)
    panels[4].set_xlabel("Time on the run clock [h]")
    for panel in panels:
        panel.set_facecolor(SURFACE)
        panel.grid(True, color=GRID, linewidth=0.6)
        panel.tick_params(colors=MUTED_TEXT, labelsize=8)
        panel.yaxis.label.set(color=TEXT, size=9)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            panel.spines[side].set_color(GRID)
        for line in panel.get_lines():
            line.set_linewidth(1.4)
    for panel in panels[:2]:
        panel.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncols=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)
    panels[4].xaxis.label.set(color=TEXT, size=9)
    figure.suptitle(run_id.replace("_", " "), color=TEXT, fontsize=11, x=0.02, ha="left")
    return figure


def main(arguments: list[str]) -> None:
    """Write one PNG per run.

    Parameters
    ----------
    arguments : list of str
        The output folder.
    """
    out_dir = Path(arguments[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in dataset.load_description(CASE)["runs"]:
        plot_run(run["id"]).savefig(out_dir / ("%s.png" % run["id"]), dpi=130)
        print("wrote", out_dir / ("%s.png" % run["id"]))


if __name__ == "__main__":
    main(sys.argv[1:])
