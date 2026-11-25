"""Readers for process-analytical-technology (PAT) files.

`read_ic_spectra` reads Mettler Toledo iC IR and iC Raman files (``.iCIR``,
``.iCRaman``) and `read_ic_fbrm` reads iC FBRM files (``.icFbrm``), without the
vendor software. The vendor does not document these formats: the layouts were
inferred from files and checked against the vendor's CSV exports of one run. A
file written by a software build outside `CHECKED_BUILDS` is read with a warning.

`read_ic_ir_export` and `read_ic_fbrm_export` read the vendor's CSV exports into
the same objects.
"""
from __future__ import annotations

import csv
import datetime
import html
import re
import struct
import warnings
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Software builds whose files were decoded and compared with the vendor's CSV exports,
# all on the caffeine-in-water run of 2026-10-23 (Nagy group, Purdue). iC IR: 663 of
# 663 spectra, largest difference 5e-15 absorbance. iC Raman: 712 of 712 spectra within
# the export's rounding (5e-10 relative). iC FBRM: 4313 samples rebinned to the 100
# export channels, median relative difference 1e-4, largest 3.3e-3, totals within 1e-5.
CHECKED_BUILDS = {"iC IR": ("7.1.91.0",), "iC Raman": ("8.0.114.0",), "iC FBRM": ("4.4.33.0",)}

# BlockPackage container of .iCIR and .iCRaman files
MAGIC = b"BlockPackage"
BLOCK_SIZE = 65536  # [byte] every stream is a chain of 64 KiB blocks
BLOCK_HEADER_SIZE = 16  # [byte] flag (uint32), used bytes (uint32), next block offset (uint64)
STREAM_HEAD_RESERVED = 2052  # [byte] name and content-type area in the first block of a stream
BLOCK_PAYLOAD = BLOCK_SIZE - BLOCK_HEADER_SIZE  # [byte] largest "used" value of a block
STREAM_HEAD_FLAG = 1  # [-] flag of the first block of a stream; later blocks carry 0
DIRECTORY_FIELD_OFFSET = 16  # [byte] position of the name-table offset and length
NAME_ENTRY_SIZE = 24  # [byte] fixed part of a name-table entry; it starts with the stream offset
FILE_INFO_PART = "/FileInfo.xml"
WRITTEN_WITH = re.compile(r"<m_(?:Created|Updated)With[^>]*>([^<]+)<")

# Spectrum records and their metadata
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
REQUIRED_ATTRIBUTES = ("SampleTime", "StartWn", "EndWn", "NumberOfPoints", "YUnits")
ABSORBANCE_UNITS = "Absorbance"  # the vendor's ``YUnits`` of iC IR spectra

# iC FBRM sample records
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
FBRM_ASSEMBLY = re.compile(rb"Fbrm, Version=([0-9.]+)")  # assembly that serialized a sample
FBRM_SOFTWARE = "iC FBRM"

# Raw grid: (number of channels, channel width [um]) from 0 um upward. Inferred from run
# 2026-10-23 by matching 4313 raw samples to the vendor's 100-channel export (median
# relative difference 1e-4, largest 3.3e-3). It is not a vendor-documented grid.
RAW_GRID_SEGMENTS = ((500, 0.2), (400, 1.0), (650, 2.0))

# Vendor CSV exports, as iC IR 7.1.91.0 and iC FBRM 4.4.33.0 wrote them for the run of
# 2026-10-23. Read back, they match the native files of that run: 663 spectra equal
# within 5e-15 absorbance with sample times within 1 s; 4313 FBRM samples as stated
# for `CHECKED_BUILDS`, with sample times cut to whole seconds.
IR_EXPORT_AXIS_TITLE = "Wavenumbercm-1"  # first cell of an iC IR export
ELAPSED_TITLE = re.compile(r"(\d+):([0-5]\d):([0-5]\d)")  # hours, minutes and seconds since the experiment started
FBRM_EXPORT_UNWEIGHTED = "No Weight"  # first cell of an iC FBRM export of unweighted counts
FBRM_EXPORT_LABELS = ("Ch.#", "Ch.Bd", "Ch. Midpoints")  # first cells of rows two to four
FBRM_EXPORT_HEADER_ROWS = 4  # [-] weighting, channel numbers, channel boundaries, channel midpoints
FBRM_EXPORT_BOUNDARIES_ROW = 2  # [-] index of the row of channel boundaries
FBRM_EXPORT_LEAD_CELLS = 4  # [-] cells before the channels: sample number, date, time, an empty cell
FBRM_EXPORT_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"  # date cell, a space, time cell: 10/24/2026 1:53:23 AM


@dataclass(frozen=True, eq=False)
class ICSpectra:
    """Spectra of one iC IR or iC Raman file, in order of sample time.

    Attributes
    ----------
    time : numpy.ndarray
        Sample times in UTC, ``datetime64[us]``, shape ``(n_spectra,)``, increasing.
        From a CSV export they are whole seconds after the start the caller gave.
    wavenumber : numpy.ndarray
        Spectral axis [cm-1], shape ``(n_points,)``, in the order the file stores it
        (decreasing in the files seen). Wavenumber for IR, Raman shift for Raman.
    values : numpy.ndarray
        Ordinates, shape ``(n_spectra, n_points)``, in ``value_units``.
    value_units : str
        The vendor's name of the ordinate: ``"Absorbance"`` [-] for iC IR,
        ``"Intensity"`` for iC Raman.
    names : tuple of str
        The vendor's name of each spectrum, in the order of ``time``.
    without_time : tuple of str
        Identifiers of the spectra the file holds without a metadata entry. They
        have no sample time and are not in ``values``.
    software : str
        Software that wrote the file, for example ``"iC IR"``; empty when unknown,
        as for a CSV export.
    build : str
        Its build, for example ``"7.1.91.0"``; empty when unknown.
    """

    time: np.ndarray
    wavenumber: np.ndarray
    values: np.ndarray
    value_units: str
    names: tuple[str, ...]
    without_time: tuple[str, ...] = ()
    software: str = ""
    build: str = ""

    @property
    def absorbance(self) -> np.ndarray:
        """Return the spectra as absorbance.

        Returns
        -------
        numpy.ndarray
            Absorbance [-], shape ``(n_spectra, n_points)``.

        Raises
        ------
        ValueError
            If the file reports another ordinate, as an iC Raman file does.
        """
        if self.value_units != ABSORBANCE_UNITS:
            raise ValueError("These spectra hold %r, not absorbance; read them from 'values'." % self.value_units)
        return self.values


@dataclass(frozen=True, eq=False)
class FBRMSamples:
    """Chord-length samples of one iC FBRM file.

    Attributes
    ----------
    time : numpy.ndarray
        Sample times in the local time of the acquisition computer,
        ``datetime64[us]``, shape ``(n_samples,)``. Whole seconds from a CSV export.
    channel_edges : numpy.ndarray
        Edges of the chord-length channels [um], shape ``(n_channels + 1,)``: the
        raw grid of `raw_channel_edges` from a native file, the boundaries the
        file states from a CSV export.
    counts : numpy.ndarray
        Unweighted chord counts [counts per channel], shape
        ``(n_samples, n_channels)``.
    software : str
        Software that wrote the file, ``"iC FBRM"``; empty when unknown, as for a
        CSV export.
    build : str
        Its build, for example ``"4.4.33.0"``; empty when unknown.
    """

    time: np.ndarray
    channel_edges: np.ndarray
    counts: np.ndarray
    software: str = ""
    build: str = ""


def _warn_unless_checked(software: str, builds: tuple[str, ...], source: object) -> None:
    """Warn when a file was written by a software build that was not checked.

    Parameters
    ----------
    software : str
        Software named by the file; empty when the file names none.
    builds : tuple of str
        Builds named by the file.
    source : object
        The file, for the message.
    """
    checked = CHECKED_BUILDS.get(software, ())
    if builds and all(build in checked for build in builds):
        return
    named = "%s build %s" % (software, ", ".join(builds)) if software and builds else "an unnamed software build"
    warnings.warn("%s was written by %s. This reader was checked against vendor exports only for %s; compare one "
                  "export of this file before relying on the values."
                  % (source, named, "; ".join("%s %s" % (name, ", ".join(value)) for name, value in CHECKED_BUILDS.items())),
                  UserWarning, stacklevel=3)


def _read_name_and_type(data: bytes, position: int) -> tuple[str, str, int]:
    """Read a part name and its content type.

    Parameters
    ----------
    data : bytes
        File contents.
    position : int
        Offset of the first of two strings, each prefixed by a one-byte length [byte].

    Returns
    -------
    tuple of (str, str, int)
        Part name, content type, and the offset just past both strings [byte].
    """
    length = data[position]
    name = data[position + 1: position + 1 + length].decode("utf-8", "replace")
    position += 1 + length
    length = data[position]
    content_type = data[position + 1: position + 1 + length].decode("utf-8", "replace")
    return name, content_type, position + 1 + length


def _read_chain(data: bytes, offset: int) -> bytes:
    """Concatenate the payload of a block chain.

    Parameters
    ----------
    data : bytes
        File contents.
    offset : int
        Offset of the first block of the stream [byte].

    Returns
    -------
    bytes
        Stream contents. The chain stops at the first block that lies outside the
        file, so a truncated file yields the part that is present.
    """
    chunks, first, visited = [], True, set()
    while offset and offset not in visited and offset % BLOCK_SIZE == 0 and offset + BLOCK_HEADER_SIZE <= len(data):
        visited.add(offset)
        _, used, next_offset = struct.unpack_from("<IIQ", data, offset)
        if used > BLOCK_PAYLOAD:
            break
        start = offset + BLOCK_HEADER_SIZE + (STREAM_HEAD_RESERVED if first else 0)
        chunks.append(data[start: start + used])
        first = False
        offset = next_offset
    return b"".join(chunks)


def is_finalized(data: bytes) -> bool:
    """Tell whether a BlockPackage file carries its name table.

    Parameters
    ----------
    data : bytes
        File contents.

    Returns
    -------
    bool
        False for a file the acquisition software never closed; its name-table
        offset and length are zero.
    """
    table_offset, table_length = struct.unpack_from("<QI", data, DIRECTORY_FIELD_OFFSET)
    return bool(table_offset and table_length)


def read_package(data: bytes) -> dict[str, tuple[str, bytes]]:
    """Return the streams of a BlockPackage file (``.iCIR``, ``.iCRaman``).

    Parameters
    ----------
    data : bytes
        File contents.

    Returns
    -------
    dict
        ``{part name: (content type, stream bytes)}``.

    Raises
    ------
    ValueError
        If ``data`` does not start with the BlockPackage signature.

    Notes
    -----
    A finalized file is read through its name table. A file without one is
    recovered by scanning every block for a stream head whose name starts with
    ``/`` and whose content type contains ``application/``.
    """
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError("Not a BlockPackage file: the signature %r is missing." % MAGIC)
    parts = {}
    if is_finalized(data):
        table_offset, _ = struct.unpack_from("<QI", data, DIRECTORY_FIELD_OFFSET)
        (part_count,) = struct.unpack_from("<I", data, table_offset)
        position = table_offset + 4
        for _ in range(part_count):
            name, content_type, position = _read_name_and_type(data, position)
            (stream_offset,) = struct.unpack_from("<Q", data, position)
            position += NAME_ENTRY_SIZE
            parts[name] = (content_type, _read_chain(data, stream_offset))
        return parts
    for offset in range(BLOCK_SIZE, len(data) - BLOCK_HEADER_SIZE + 1, BLOCK_SIZE):
        flag, used, _ = struct.unpack_from("<IIQ", data, offset)
        position = offset + BLOCK_HEADER_SIZE
        if flag != STREAM_HEAD_FLAG or used > BLOCK_PAYLOAD or data[position + 1: position + 2] != b"/":
            continue
        name, content_type, _ = _read_name_and_type(data, position)
        if "application/" in content_type:
            parts[name] = (content_type, _read_chain(data, offset))
    return parts


def _written_with(file_info: bytes) -> tuple[str, tuple[str, ...]]:
    """Read the software and builds named in a ``/FileInfo.xml`` part.

    Parameters
    ----------
    file_info : bytes
        Contents of the part; its ``m_CreatedWith`` and ``m_UpdatedWith`` entries
        read like ``iC IR`` followed by ``Build: 7.1.91.0`` on the next line.

    Returns
    -------
    tuple of (str, tuple of str)
        Software of the first entry, and the distinct builds of all entries in
        document order. Empty when the part names none.
    """
    software, builds = "", []
    for entry in WRITTEN_WITH.findall(file_info.decode("utf-8", "replace")):
        name, _, build = html.unescape(entry).partition("Build:")
        software = software or name.strip()
        if build.strip() and build.strip() not in builds:
            builds.append(build.strip())
    return software, tuple(builds)


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
        units named by the matching metadata entry.
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
        Text of the ``/SpectraExperiment<n>.xml`` part.

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


def _parse_utc(stamp: str) -> datetime.datetime:
    """Parse an iC ``SampleTime`` such as ``2026-10-24T04:29:10.2574757Z``.

    Parameters
    ----------
    stamp : str
        ISO 8601 UTC time with up to seven fractional digits.

    Returns
    -------
    datetime.datetime
        Naive UTC time, truncated to microseconds.
    """
    head, _, fraction = stamp.rstrip("Z").partition(".")
    return datetime.datetime.strptime(head + "." + (fraction[:6] or "0"), "%Y-%m-%dT%H:%M:%S.%f")


def spectra_from_records(infos: dict[str, dict[str, str]], records: dict[str, np.ndarray], software: str = "",
                         build: str = "") -> ICSpectra:
    """Join spectrum records with their metadata into arrays.

    Parameters
    ----------
    infos : dict
        Metadata from `spectrum_infos`.
    records : dict
        ``{spectrum identifier: values}`` from `spectrum_records`, each of shape
        ``(n_points,)``.
    software, build : str, optional
        Software that wrote the records and its build, stored in the result.

    Returns
    -------
    ICSpectra
        The spectra that have a metadata entry, sorted by sample time. The
        identifiers of the others are in ``without_time``.

    Raises
    ------
    ValueError
        If no spectrum has a metadata entry, an entry lacks an attribute of
        `REQUIRED_ATTRIBUTES`, the spectra do not share one spectral axis and one
        ordinate, or a record does not have the number of points of its entry.
    """
    rows, axes, units, without_time = [], set(), set(), []
    for identifier, values in records.items():
        if identifier not in infos:
            without_time.append(identifier)
            continue
        info = infos[identifier]
        missing = [key for key in REQUIRED_ATTRIBUTES if key not in info]
        if missing:
            raise ValueError("The metadata entry of spectrum %s lacks %s." % (identifier, ", ".join(missing)))
        axis = (float(info["StartWn"]), float(info["EndWn"]), int(info["NumberOfPoints"]))
        if values.size != axis[2]:
            raise ValueError("Spectrum %s holds %d values but its metadata entry declares %d points."
                             % (identifier, values.size, axis[2]))
        axes.add(axis)
        units.add(info["YUnits"])
        rows.append((_parse_utc(info["SampleTime"]), info.get("Name", ""), values))
    if not rows:
        raise ValueError("No spectrum has a metadata entry, so none has a sample time.")
    if len(axes) != 1:
        raise ValueError("The spectra do not share one spectral axis: %s." % sorted(axes))
    if len(units) != 1:
        raise ValueError("The spectra do not share one ordinate: %s." % sorted(units))
    start, end, points = axes.pop()
    rows.sort(key=lambda row: row[0])
    return ICSpectra(
        time=np.array([row[0] for row in rows], dtype="datetime64[us]"),
        wavenumber=np.linspace(start, end, points),  # [cm-1]
        values=np.vstack([row[2] for row in rows]),
        value_units=units.pop(), names=tuple(row[1] for row in rows), without_time=tuple(without_time),
        software=software, build=build)


def read_ic_spectra(path: str | Path) -> ICSpectra:
    """Read every spectrum of an iC IR or iC Raman file (``.iCIR``, ``.iCRaman``).

    Parameters
    ----------
    path : str or pathlib.Path
        Native file.

    Returns
    -------
    ICSpectra
        Sample times (UTC), spectral axis [cm-1] and ordinates of the spectra,
        sorted by sample time.

    Raises
    ------
    ValueError
        If the file is not a BlockPackage file, does not hold exactly one
        experiment document, that document is empty, or the spectra cannot be
        joined with their metadata (see `spectra_from_records`).

    Warns
    -----
    UserWarning
        If the file was written by a software build outside `CHECKED_BUILDS`.

    Examples
    --------
    >>> ir = read_ic_spectra("run.iCIR")  # doctest: +SKIP
    >>> ir.time.shape, ir.wavenumber.shape, ir.absorbance.shape  # doctest: +SKIP
    ((864,), (589,), (864, 589))
    """
    parts = read_package(Path(path).read_bytes())
    experiment_names = [name for name in parts if EXPERIMENT_PART.fullmatch(name)]
    if len(experiment_names) != 1:
        raise ValueError("Expected one /SpectraExperiment*.xml part in %s, found %d." % (path, len(experiment_names)))
    raw = parts[experiment_names[0]][1]
    if len(raw) < LENGTH_FIELD_SIZE:
        raise ValueError("The experiment document of %s is empty; the file is incomplete." % path)
    (length,) = struct.unpack_from("<I", raw, 0)
    experiment_xml = raw[LENGTH_FIELD_SIZE: LENGTH_FIELD_SIZE + length].decode("utf-8", "replace")
    records = {}
    for name, (_, stream) in parts.items():
        if name.endswith("Spectras.vector"):
            records.update(spectrum_records(stream))
    software, builds = _written_with(parts.get(FILE_INFO_PART, ("", b""))[1])
    _warn_unless_checked(software, builds, path)
    return spectra_from_records(spectrum_infos(experiment_xml), records, software, builds[0] if builds else "")


def raw_channel_edges() -> np.ndarray:
    """Return the edges of the raw chord-length channels of iC FBRM.

    Returns
    -------
    numpy.ndarray
        Channel edges [um], shape ``(1551,)``, increasing from 0 to 1800 um.

    Notes
    -----
    The grid is `RAW_GRID_SEGMENTS`: inferred from one run, not documented by the
    vendor.
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


def read_ic_fbrm(path: str | Path) -> FBRMSamples:
    """Read every chord-length sample of an iC FBRM file (``.icFbrm``).

    Parameters
    ----------
    path : str or pathlib.Path
        Native file (a zip archive).

    Returns
    -------
    FBRMSamples
        Sample times (local time of the acquisition computer), channel edges [um]
        and unweighted chord counts [counts per channel] on the raw grid.

    Raises
    ------
    ValueError
        If the archive has no ``ReactionSamples.vector`` part, that part holds no
        sample record, or the samples are not on the grid of `raw_channel_edges`.

    Warns
    -----
    UserWarning
        If the samples were written by a software build outside `CHECKED_BUILDS`.

    Examples
    --------
    >>> fbrm = read_ic_fbrm("run.icFbrm")  # doctest: +SKIP
    >>> fbrm.time.shape, fbrm.channel_edges.shape, fbrm.counts.shape  # doctest: +SKIP
    ((5393,), (1551,), (5393, 1550))
    """
    with zipfile.ZipFile(path) as archive:
        if SAMPLES_PART not in archive.namelist():
            raise ValueError("%s holds no %s part." % (path, SAMPLES_PART))
        vector = archive.read(SAMPLES_PART)
    first = FIRST_RECORD.search(vector, RECORD_SEARCH_START)
    if first is None or len(vector) < RECORDS_END_FIELD + LENGTH_FIELD_SIZE:
        raise ValueError("No FBRM sample record in %s." % path)
    (records_end,) = struct.unpack_from("<I", vector, RECORDS_END_FIELD)
    position = first.start()
    times, counts, builds = [], [], []
    while position + RECORD_PREFIX_SIZE <= records_end and vector[position: position + 2] == RECORD_MARK:
        (ticks,) = struct.unpack_from("<q", vector, position + TICKS_FIELD)
        (length,) = struct.unpack_from("<I", vector, position + PAYLOAD_LENGTH_FIELD)
        payload = vector[position + RECORD_PREFIX_SIZE: position + RECORD_PREFIX_SIZE + length]
        if not times:
            builds = [match.decode("ascii") for match in FBRM_ASSEMBLY.findall(payload)[:1]]
        times.append(DOTNET_EPOCH + datetime.timedelta(microseconds=(ticks & DATETIME_KIND_MASK) // TICKS_PER_MICROSECOND))
        counts.append(_largest_double_array(payload))
        position += RECORD_PREFIX_SIZE + length
    edges = raw_channel_edges()  # [um]
    sizes = sorted({sample.size for sample in counts})
    if sizes != [edges.size - 1]:
        raise ValueError("%s holds samples of %s channels; the raw grid of this reader has %d."
                         % (path, sizes, edges.size - 1))
    software = FBRM_SOFTWARE if builds else ""
    _warn_unless_checked(software, tuple(builds), path)
    return FBRMSamples(time=np.array(times, dtype="datetime64[us]"), channel_edges=edges, counts=np.array(counts),
                       software=software, build=builds[0] if builds else "")


def _export_rows(path: str | Path) -> list[list[str]]:
    """Read the rows of a vendor CSV export.

    Parameters
    ----------
    path : str or pathlib.Path
        Comma-separated text file.

    Returns
    -------
    list of list of str
        The cells of every line that holds one, without surrounding blanks and
        without the empty cells that end a line.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [[cell.strip() for cell in row] for row in csv.reader(handle)]
    for row in rows:
        while row and not row[-1]:
            row.pop()
    return [row for row in rows if row]


def _numbers(rows: list[list[str]], path: str | Path, what: str) -> np.ndarray:
    """Convert rows of cells of equal length to finite numbers.

    Parameters
    ----------
    rows : list of list of str
        Cells, one list per row.
    path : str or pathlib.Path
        The file, for the message.
    what : str
        Name of the values, for the message.

    Returns
    -------
    numpy.ndarray
        The values in the unit of the file, shape ``(n_rows, n_cells)``.

    Raises
    ------
    ValueError
        If a cell is empty, not a number, or not finite.
    """
    try:
        values = np.array([[float(cell) for cell in row] for row in rows])
    except ValueError:
        raise ValueError("%s of %s hold a cell that is not a number." % (what, path)) from None
    if not np.all(np.isfinite(values)):
        raise ValueError("%s of %s hold a value that is not finite." % (what, path))
    return values


def read_ic_ir_export(path: str | Path, start_utc: datetime.datetime) -> ICSpectra:
    """Read every spectrum of an iC IR CSV export.

    Parameters
    ----------
    path : str or pathlib.Path
        Export with one row per wavenumber and one column per spectrum. The first
        cell is ``Wavenumbercm-1``; every other column title is the time since the
        experiment started, written as ``hh:mm:ss``.
    start_utc : datetime.datetime
        Time at which the iC IR experiment started, in UTC. The export holds no
        date, so the sample times are this start plus the column titles. A
        timezone-aware value is converted to UTC.

    Returns
    -------
    ICSpectra
        Sample times (UTC, whole seconds), wavenumber [cm-1] and absorbance [-] of
        the spectra, in the order of the columns. ``names`` holds the column
        titles; ``software`` and ``build`` are empty.

    Raises
    ------
    TypeError
        If ``start_utc`` is not a ``datetime.datetime``.
    ValueError
        If the file is not an iC IR export, holds no spectrum, a column title is
        not ``hh:mm:ss``, the titles decrease, a row has another number of cells
        than the title row, or a cell is not a finite number.

    Notes
    -----
    The export does not name its ordinate. It is taken as absorbance, which the
    export of `CHECKED_BUILDS` holds; an export made in another ordinate is not
    detected. A title beyond 24 hours was never seen: it is read if the hours
    keep counting and refused otherwise.

    Examples
    --------
    >>> start = datetime.datetime(2026, 10, 24, 4, 28, 19)  # doctest: +SKIP
    >>> ir = read_ic_ir_export("ir.csv", start)  # doctest: +SKIP
    >>> ir.time.shape, ir.wavenumber.shape, ir.absorbance.shape  # doctest: +SKIP
    ((663,), (589,), (663, 589))
    """
    if not isinstance(start_utc, datetime.datetime):
        raise TypeError("start_utc must be a datetime.datetime, not %s." % type(start_utc).__name__)
    if start_utc.tzinfo is not None:
        start_utc = start_utc.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    rows = _export_rows(path)
    if not rows or rows[0][0] != IR_EXPORT_AXIS_TITLE:
        raise ValueError("%s is not an iC IR export: its first cell must be %r." % (path, IR_EXPORT_AXIS_TITLE))
    titles = rows[0][1:]
    if not titles or len(rows) < 2:
        raise ValueError("%s holds no spectrum." % path)
    elapsed = []
    for title in titles:
        match = ELAPSED_TITLE.fullmatch(title)
        if match is None:
            raise ValueError("Column title %r of %s is not a time since the start of the experiment written as hh:mm:ss."
                             % (title, path))
        hours, minutes, seconds = (int(group) for group in match.groups())
        elapsed.append(datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds))
    if any(later < earlier for earlier, later in zip(elapsed, elapsed[1:])):
        raise ValueError("The column titles of %s do not increase, so they do not give the sample times." % path)
    for number, row in enumerate(rows[1:], start=1):
        if len(row) != len(rows[0]):
            raise ValueError("Data row %d of %s has %d cells; the title row has %d." % (number, path, len(row), len(rows[0])))
    table = _numbers(rows[1:], path, "The spectra")
    return ICSpectra(
        time=np.array([start_utc + delay for delay in elapsed], dtype="datetime64[us]"),
        wavenumber=table[:, 0].copy(),  # [cm-1]
        values=np.ascontiguousarray(table[:, 1:].T),  # [-] absorbance, shape (n_spectra, n_points)
        value_units=ABSORBANCE_UNITS, names=tuple(titles))


def read_ic_fbrm_export(path: str | Path) -> FBRMSamples:
    """Read every chord-length sample of an iC FBRM CSV export.

    Parameters
    ----------
    path : str or pathlib.Path
        Export with four header rows (weighting, ``Ch.#``, ``Ch.Bd``,
        ``Ch. Midpoints``) and one row per sample: sample number, date, time, an
        empty cell and the counts per channel.

    Returns
    -------
    FBRMSamples
        Sample times (local time of the acquisition computer, whole seconds), the
        channel boundaries the file states [um] and unweighted chord counts
        [counts per channel], in the order of the rows. ``software`` and ``build``
        are empty.

    Raises
    ------
    ValueError
        If the file is not an iC FBRM export, its counts are weighted, the channel
        boundaries do not increase, it holds no sample, a sample does not have one
        count per channel, a date and time are not written like
        ``10/24/2026 1:53:23 AM``, or a cell is not a finite number.

    Notes
    -----
    The boundaries are read as printed, to five significant digits in the export
    of `CHECKED_BUILDS`. The export has fewer channels than the native file (100
    against 1550 in that export), so counts moved to other edges with
    `rebin_matrix` are split inside a channel in proportion to its length.

    Examples
    --------
    >>> fbrm = read_ic_fbrm_export("fbrm.csv")  # doctest: +SKIP
    >>> fbrm.time.shape, fbrm.channel_edges.shape, fbrm.counts.shape  # doctest: +SKIP
    ((4313,), (101,), (4313, 100))
    """
    rows = _export_rows(path)
    if len(rows) < FBRM_EXPORT_HEADER_ROWS or tuple(row[0] for row in rows[1:FBRM_EXPORT_HEADER_ROWS]) != FBRM_EXPORT_LABELS:
        raise ValueError("%s is not an iC FBRM export: its rows two to four must start with %s."
                         % (path, ", ".join(FBRM_EXPORT_LABELS)))
    if rows[0][0] != FBRM_EXPORT_UNWEIGHTED:
        raise ValueError("%s holds counts with the weighting %r; only an export made with %r is read."
                         % (path, rows[0][0], FBRM_EXPORT_UNWEIGHTED))
    boundaries = rows[FBRM_EXPORT_BOUNDARIES_ROW][FBRM_EXPORT_LEAD_CELLS:]
    edges = _numbers([boundaries], path, "The channel boundaries")[0]  # [um]
    if edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("The channel boundaries of %s must be at least two increasing values." % path)
    samples = rows[FBRM_EXPORT_HEADER_ROWS:]
    if not samples:
        raise ValueError("%s holds no FBRM sample." % path)
    times = []
    for row in samples:
        if len(row) != FBRM_EXPORT_LEAD_CELLS + edges.size - 1:
            raise ValueError("Sample %s of %s has %d counts; the file states %d channels."
                             % (row[0], path, max(len(row) - FBRM_EXPORT_LEAD_CELLS, 0), edges.size - 1))
        try:
            times.append(datetime.datetime.strptime("%s %s" % (row[1], row[2]), FBRM_EXPORT_TIME_FORMAT))
        except ValueError:
            raise ValueError("Sample %s of %s has the date and time %r, not written like 10/24/2026 1:53:23 AM."
                             % (row[0], path, "%s %s" % (row[1], row[2]))) from None
    counts = _numbers([row[FBRM_EXPORT_LEAD_CELLS:] for row in samples], path, "The counts")  # [counts per channel]
    return FBRMSamples(time=np.array(times, dtype="datetime64[us]"), channel_edges=edges, counts=counts)
