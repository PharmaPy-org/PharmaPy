"""Decode chord-length samples from Mettler Toledo iC FBRM native files (``.icFbrm``).

The record layout and the raw channel grid were inferred from the caffeine runs and
checked against the vendor's CSV export; see ``README.md`` in this directory.
"""
from __future__ import annotations

import datetime
import re
import struct
import zipfile
from pathlib import Path

import numpy as np

SAMPLES_PART = "ReactionSamples.vector"
RECORD_PREFIX_SIZE = 35  # [byte] marker (2), ticks (8), 5 bytes, payload length (4), identifier (16)
RECORD_MARK = b"\xfe\xfe"
FIRST_RECORD = re.compile(rb"\xfe\xfe.{8}\x05", re.S)
RECORD_SEARCH_START = 128  # [byte] the stream header is longer than this
RECORDS_END_FIELD = 164  # [byte] position of the uint32 offset where the records end
TICKS_FIELD = 2  # [byte] within a record: int64 .NET ticks of the sample time
PAYLOAD_LENGTH_FIELD = 15  # [byte] within a record: uint32 payload length
DOTNET_EPOCH = datetime.datetime(1, 1, 1)
TICKS_PER_MICROSECOND = 10  # [ticks/us] one .NET tick is 100 ns
DATETIME_KIND_MASK = 0x3FFFFFFFFFFFFFFF  # the two top bits of a .NET DateTime carry its Kind
PRIMITIVE_ARRAY_RECORD = 0x0F  # .NET BinaryFormatter record type ArraySinglePrimitive
PRIMITIVE_ARRAY_HEADER = 10  # [byte] record type (1), object id (4), length (4), element type (1)
DOUBLE_TYPE = 6  # .NET BinaryFormatter primitive type code of Double
MAX_ARRAY_LENGTH = 100000  # [-] sanity bound; the chord distribution has 1550 channels

# Raw grid: (number of channels, channel width [um]) from 0 um upward. Inferred from run
# 2026-10-23 by matching 4313 raw samples to the vendor's 100-channel export (median
# relative difference 1e-4, largest 3.3e-3). It is not a vendor-documented grid.
RAW_GRID_SEGMENTS = ((500, 0.2), (400, 1.0), (650, 2.0))


def raw_channel_edges() -> np.ndarray:
    """Return the edges of the raw chord-length channels.

    Returns
    -------
    numpy.ndarray
        Channel edges [um], shape ``(1551,)``, increasing from 0 to 1800 um.
    """
    edges = [np.zeros(1)]
    for count, width in RAW_GRID_SEGMENTS:
        edges.append(edges[-1][-1] + width * np.arange(1, count + 1))  # [um]
    return np.concatenate(edges)


def rebin_matrix(source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    """Build the matrix that moves counts from one channel grid to another.

    Parameters
    ----------
    source_edges : numpy.ndarray
        Edges of the source channels [um], shape ``(n_source + 1,)``, increasing.
    target_edges : numpy.ndarray
        Edges of the target channels [um], shape ``(n_target + 1,)``, increasing.

    Returns
    -------
    numpy.ndarray
        Matrix ``M`` [-] of shape ``(n_source, n_target)`` with
        ``target_counts = source_counts @ M``. Each source channel is split among
        the target channels in proportion to the overlapping length, so counts
        inside the target range are conserved and counts outside it are dropped.

    Raises
    ------
    ValueError
        If either set of edges is not one-dimensional and strictly increasing.
    """
    source_edges = np.asarray(source_edges, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)
    for name, edges in (("source_edges", source_edges), ("target_edges", target_edges)):
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("%s must be a one-dimensional, strictly increasing array of edges." % name)
    lower = np.maximum(source_edges[:-1, None], target_edges[None, :-1])  # [um]
    upper = np.minimum(source_edges[1:, None], target_edges[None, 1:])  # [um]
    return np.clip(upper - lower, 0.0, None) / np.diff(source_edges)[:, None]


def _largest_double_array(payload: bytes) -> np.ndarray:
    """Return the longest float64 array serialized in a sample payload.

    Parameters
    ----------
    payload : bytes
        .NET BinaryFormatter serialization of one ``FbrmSample``.

    Returns
    -------
    numpy.ndarray
        The chord-length distribution [counts per channel], shape ``(n_channels,)``.

    Raises
    ------
    ValueError
        If the payload holds no float64 array.
    """
    best, position = None, 0
    while True:
        start = payload.find(bytes((PRIMITIVE_ARRAY_RECORD,)), position)
        if start < 0 or start + PRIMITIVE_ARRAY_HEADER > len(payload):
            break
        object_id, length, element_type = struct.unpack_from("<iiB", payload, start + 1)
        data_start = start + PRIMITIVE_ARRAY_HEADER
        if element_type == DOUBLE_TYPE and object_id > 0 and 0 < length <= MAX_ARRAY_LENGTH and data_start + 8 * length <= len(payload):
            if best is None or length > best.size:
                best = np.frombuffer(payload, dtype="<f8", count=length, offset=data_start)
            position = data_start + 8 * length
        else:
            position = start + 1
    if best is None:
        raise ValueError("The FBRM sample payload holds no float64 array.")
    return best


def read_samples(path: str | Path) -> tuple[list[datetime.datetime], np.ndarray]:
    """Read every chord-length sample of an ``.icFbrm`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Native file (a zip archive).

    Returns
    -------
    tuple of (list of datetime.datetime, numpy.ndarray)
        Sample times as naive datetimes in the acquisition computer's local time,
        and the unweighted chord counts [counts per channel] of shape
        ``(n_samples, n_channels)`` on the grid of `raw_channel_edges`.

    Raises
    ------
    ValueError
        If the archive holds no sample record.
    """
    with zipfile.ZipFile(path) as archive:
        vector = archive.read(SAMPLES_PART)
    (records_end,) = struct.unpack_from("<I", vector, RECORDS_END_FIELD)
    first = FIRST_RECORD.search(vector, RECORD_SEARCH_START)
    if first is None:
        raise ValueError("No FBRM sample record in %s." % path)
    position = first.start()
    times, counts = [], []
    while position + RECORD_PREFIX_SIZE <= records_end and vector[position: position + 2] == RECORD_MARK:
        (ticks,) = struct.unpack_from("<q", vector, position + TICKS_FIELD)
        (length,) = struct.unpack_from("<I", vector, position + PAYLOAD_LENGTH_FIELD)
        payload = vector[position + RECORD_PREFIX_SIZE: position + RECORD_PREFIX_SIZE + length]
        times.append(DOTNET_EPOCH + datetime.timedelta(microseconds=(ticks & DATETIME_KIND_MASK) // TICKS_PER_MICROSECOND))
        counts.append(_largest_double_array(payload))
        position += RECORD_PREFIX_SIZE + length
    return times, np.array(counts)
