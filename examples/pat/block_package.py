"""Reader for the Mettler Toledo "BlockPackage" container (``.iCIR``, ``.iCRaman``).

The layout is not documented by the vendor. It was inferred from the caffeine runs
and checked against the vendor's CSV exports; see ``README.md`` in this directory.
"""
import struct

MAGIC = b"BlockPackage"
BLOCK_SIZE = 65536  # [byte] every stream is a chain of 64 KiB blocks
BLOCK_HEADER_SIZE = 16  # [byte] flag (uint32), used bytes (uint32), next block offset (uint64)
STREAM_HEAD_RESERVED = 2052  # [byte] name and content-type area in the first block of a stream
BLOCK_PAYLOAD = BLOCK_SIZE - BLOCK_HEADER_SIZE  # [byte] largest "used" value of a block
STREAM_HEAD_FLAG = 1  # [-] flag of the first block of a stream; later blocks carry 0
DIRECTORY_FIELD_OFFSET = 16  # [byte] position of the name-table offset and length
NAME_ENTRY_SIZE = 24  # [byte] fixed part of a name-table entry; it starts with the stream offset


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
    """Tell whether the file carries its name table.

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
    """Return the streams of a BlockPackage file.

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
