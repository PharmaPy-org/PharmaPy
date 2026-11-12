"""Decode spectra from Mettler Toledo iC IR and iC Raman native files.

The record layout was inferred from the caffeine runs and checked against the
vendor's CSV exports; see ``README.md`` in this directory.
"""
from __future__ import annotations

import re
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .block_package import read_package

XML_MARK = b"\xef\xbb\xbf<?xml"  # UTF-8 byte-order mark followed by the XML declaration
BYTE_ORDER_MARK_SIZE = 3  # [byte]
LENGTH_FIELD_SIZE = 4  # [byte] uint32 length that precedes the XML and the values
MAX_RECORD_XML = 4 * 1024 * 1024  # [byte] sanity bound; observed record headers are about 3 kB
MIN_RECORD_XML = 16  # [byte] shorter than any XML declaration, so not a record header
VALUE_SIZE = 8  # [byte] spectra are stored as little-endian float64
ATTRIBUTE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
RECORD_ID = re.compile(r'<SpectrumData[^>]*\sID="([^"]+)"')
INFO_TAG = re.compile(r'<(?:Info|[A-Za-z]*SpectrumInfo) __type="[A-Za-z]*SpectrumInfo"([^>]*)>')
EXPERIMENT_PART = re.compile(r"/SpectraExperiment\d*\.xml")


def spectrum_records(vector: bytes) -> Iterator[tuple[str, np.ndarray]]:
    """Yield the spectra stored in a ``Spectras.vector`` stream.

    Parameters
    ----------
    vector : bytes
        Stream contents. Each record is a uint32 XML length, a UTF-8 XML header
        ``<SpectrumData ID=...>`` with a byte-order mark, a uint32 byte count and
        that many bytes of float64 values.

    Yields
    ------
    tuple of (str, numpy.ndarray)
        Spectrum identifier and its ordinate values, shape ``(n_points,)``, in the
        units named by the matching metadata entry (absorbance [-] for iC IR,
        intensity [counts] for iC Raman).
    """
    position = 0
    while True:
        start = vector.find(XML_MARK, position)
        if start < LENGTH_FIELD_SIZE:
            if start < 0:
                return
            position = start + len(XML_MARK)
            continue
        position = start + len(XML_MARK)
        (xml_length,) = struct.unpack_from("<I", vector, start - LENGTH_FIELD_SIZE)
        end = start + xml_length
        if not MIN_RECORD_XML <= xml_length <= MAX_RECORD_XML or end + LENGTH_FIELD_SIZE > len(vector):
            continue
        header = vector[start + BYTE_ORDER_MARK_SIZE: end].decode("utf-8", "replace")
        (byte_count,) = struct.unpack_from("<I", vector, end)
        identifier = RECORD_ID.search(header)
        data_start = end + LENGTH_FIELD_SIZE
        if identifier is None or byte_count == 0 or byte_count % VALUE_SIZE or data_start + byte_count > len(vector):
            continue
        yield identifier.group(1), np.frombuffer(vector, dtype="<f8", count=byte_count // VALUE_SIZE, offset=data_start)
        position = data_start + byte_count


def spectrum_infos(experiment_xml: str) -> dict[str, dict[str, str]]:
    """Collect the per-spectrum metadata of an experiment document.

    Parameters
    ----------
    experiment_xml : str
        Text of ``/SpectraExperiment0.xml``.

    Returns
    -------
    dict
        ``{spectrum identifier: attributes}`` in document order. Attributes are the
        vendor's strings, for example ``Name``, ``SampleTime`` (UTC, ISO 8601),
        ``StartWn`` and ``EndWn`` [cm-1], ``NumberOfPoints`` [-], ``YUnits`` and,
        for Raman, ``ExposureTime`` (ISO 8601 duration).
    """
    infos = {}
    for match in INFO_TAG.finditer(experiment_xml):
        attributes = dict(ATTRIBUTE.findall(match.group(1)))
        if "ID" in attributes:
            infos[attributes["ID"]] = attributes
    return infos


def load_spectra(path: str | Path) -> tuple[dict[str, dict[str, str]], dict[str, np.ndarray]]:
    """Read every spectrum and its metadata from an ``.iCIR`` or ``.iCRaman`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Native file.

    Returns
    -------
    tuple of (dict, dict)
        Metadata as returned by `spectrum_infos`, and ``{spectrum identifier:
        values}`` with each array of shape ``(n_points,)``.

    Raises
    ------
    ValueError
        If the file has no experiment document.
    """
    parts = read_package(Path(path).read_bytes())
    experiment_names = [name for name in parts if EXPERIMENT_PART.fullmatch(name)]
    if not experiment_names:
        raise ValueError("No /SpectraExperiment*.xml part in %s." % path)
    raw = parts[experiment_names[0]][1]
    (length,) = struct.unpack_from("<I", raw, 0)
    experiment_xml = raw[LENGTH_FIELD_SIZE: LENGTH_FIELD_SIZE + length].decode("utf-8", "replace")
    records = {}
    for name, (_, stream) in parts.items():
        if name.endswith("Spectras.vector"):
            records.update(spectrum_records(stream))
    return spectrum_infos(experiment_xml), records
