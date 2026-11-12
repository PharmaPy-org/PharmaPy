"""Common table of one experimental run: named series on a single run clock.

Instruments sample at different rates and start at different moments. Each series
keeps its own sample times, shifted onto the clock of the run, so later steps read
the same structure whichever file type the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def block_mean(values: np.ndarray, size: int) -> np.ndarray:
    """Average consecutive blocks of samples along the first axis.

    Parameters
    ----------
    values : numpy.ndarray
        Samples, shape ``(n_samples,)`` or ``(n_samples, n_columns)``.
    size : int
        Number of samples per block [-]; at least 1.

    Returns
    -------
    numpy.ndarray
        Block means in the unit of ``values``, shape ``(n_samples // size, ...)``.
        Samples after the last complete block are dropped.

    Raises
    ------
    ValueError
        If ``size`` is below 1 or exceeds the number of samples.
    """
    values = np.asarray(values, dtype=float)
    if size < 1 or size > values.shape[0]:
        raise ValueError("Block size %d does not fit %d samples." % (size, values.shape[0]))
    blocks = values.shape[0] // size
    return values[: blocks * size].reshape((blocks, size) + values.shape[1:]).mean(axis=1)


@dataclass(frozen=True)
class Series:
    """One measured quantity of a run.

    Attributes
    ----------
    name : str
        Field name.
    unit : str
        Unit of ``values``; ``"-"`` when dimensionless.
    time : numpy.ndarray
        Sample times on the run clock [s], shape ``(n_samples,)``, strictly increasing.
    values : numpy.ndarray
        Samples, shape ``(n_samples,)`` or ``(n_samples, n_columns)``.
    """

    name: str
    unit: str
    time: np.ndarray
    values: np.ndarray


class RunTable:
    """Series of one run on a shared clock.

    The run clock is whatever the case declares as time zero, for example the start
    of the temperature program. Nothing is interpolated or resampled.
    """

    def __init__(self, run_id: str):
        """Create an empty table.

        Parameters
        ----------
        run_id : str
            Identifier of the run.
        """
        self.run_id = run_id
        self.series: dict[str, Series] = {}

    def add(self, name: str, unit: str, time: np.ndarray, values: np.ndarray, clock_offset: float = 0.0) -> Series:
        """Add a series, shifting its times onto the run clock.

        Parameters
        ----------
        name : str
            Field name; must be new to the table.
        unit : str
            Unit of ``values``.
        time : numpy.ndarray
            Sample times on the instrument's own clock [s], shape ``(n_samples,)``.
        values : numpy.ndarray
            Samples, shape ``(n_samples,)`` or ``(n_samples, n_columns)``.
        clock_offset : float, optional
            Run-clock time of the instrument's time zero [s]. Default 0.

        Returns
        -------
        Series
            The stored series, with ``time + clock_offset`` as its times [s].

        Raises
        ------
        ValueError
            If the name is taken, the shapes disagree, a value is not finite, or
            the times do not increase strictly.
        """
        time = np.asarray(time, dtype=float)
        values = np.asarray(values, dtype=float)
        if name in self.series:
            raise ValueError("Run %s already has a series named %r." % (self.run_id, name))
        if time.ndim != 1 or values.ndim not in (1, 2) or values.shape[0] != time.size:
            raise ValueError("Series %r: time has shape %s but values have shape %s; the first axes must match."
                             % (name, time.shape, values.shape))
        if not (np.all(np.isfinite(time)) and np.all(np.isfinite(values)) and np.isfinite(clock_offset)):
            raise ValueError("Series %r holds a non-finite time, value or clock offset." % name)
        if not np.all(np.diff(time) > 0):
            raise ValueError("Series %r: times do not increase strictly." % name)
        self.series[name] = Series(name=name, unit=unit, time=time + clock_offset, values=values)
        return self.series[name]
