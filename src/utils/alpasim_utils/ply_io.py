# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Python+numpy PLY I/O for triangle meshes (vertices + faces).
Supports ASCII, binary little-endian, and binary big-endian PLY formats.
"""

import io
import struct
from typing import Literal

import numpy as np

# PLY type name -> (struct format char, numpy dtype)
_PLY_TYPES: dict[str, tuple[str, type[np.generic]]] = {
    "char": ("b", np.int8),
    "uchar": ("B", np.uint8),
    "short": ("h", np.int16),
    "ushort": ("H", np.uint16),
    "int": ("i", np.int32),
    "uint": ("I", np.uint32),
    "float": ("f", np.float32),
    "double": ("d", np.float64),
    # Alternate numeric names used by some exporters
    "int8": ("b", np.int8),
    "uint8": ("B", np.uint8),
    "int16": ("h", np.int16),
    "uint16": ("H", np.uint16),
    "int32": ("i", np.int32),
    "uint32": ("I", np.uint32),
    "float32": ("f", np.float32),
    "float64": ("d", np.float64),
}

_Property = tuple  # ("scalar", type_name, prop_name) | ("list", count_type, index_type, prop_name)
_Element = dict  # {"name": str, "count": int, "properties": list[_Property]}
_Format = Literal["ascii", "binary_little_endian", "binary_big_endian"]


def _parse_header(stream: io.BytesIO) -> tuple[_Format, list[_Element]]:
    """Parse a PLY header and return (format_string, elements)."""
    magic = stream.readline().strip()
    if magic != b"ply":
        raise ValueError(f"Not a PLY file (got {magic!r})")

    fmt: _Format | None = None
    elements: list[_Element] = []
    current_element: _Element | None = None

    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Unexpected end of file while reading PLY header")

        tokens = line.strip().decode("ascii").split()
        if not tokens or tokens[0] == "comment" or tokens[0] == "obj_info":
            continue

        if tokens[0] == "end_header":
            break
        elif tokens[0] == "format":
            _VALID_FORMATS = {"ascii", "binary_little_endian", "binary_big_endian"}
            if tokens[1] not in _VALID_FORMATS:
                raise ValueError(
                    f"Unsupported PLY format: {tokens[1]!r} "
                    f"(expected one of {sorted(_VALID_FORMATS)})"
                )
            fmt = tokens[1]  # type: ignore[assignment]
        elif tokens[0] == "element":
            if current_element is not None:
                elements.append(current_element)
            current_element = {
                "name": tokens[1],
                "count": int(tokens[2]),
                "properties": [],
            }
        elif tokens[0] == "property":
            if current_element is None:
                raise ValueError("Unexpected 'property' line in header")
            if tokens[1] == "list":
                current_element["properties"].append(
                    ("list", tokens[2], tokens[3], tokens[4])
                )
            else:
                current_element["properties"].append(("scalar", tokens[1], tokens[2]))

    if current_element is not None:
        elements.append(current_element)
    if fmt is None:
        raise ValueError("PLY header missing 'format' line")

    return fmt, elements


def _find_elements(
    elements: list[_Element],
) -> tuple[_Element, _Element]:
    """Return the vertex and face elements, raising if either is missing."""
    vertex_elem: _Element | None = None
    face_elem: _Element | None = None
    for elem in elements:
        if elem["name"] == "vertex":
            vertex_elem = elem
        elif elem["name"] == "face":
            face_elem = elem
    if vertex_elem is None:
        raise ValueError("PLY file has no 'vertex' element")
    if face_elem is None:
        raise ValueError("PLY file has no 'face' element")
    return vertex_elem, face_elem


# ---------------------------------------------------------------------------
# Binary reading
# ---------------------------------------------------------------------------

_Endian = Literal["<", ">"]


def _load_binary(
    stream: io.BytesIO,
    elements: list[_Element],
    vertex_elem: _Element,
    face_elem: _Element,
    endian: _Endian,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None

    for elem in elements:
        if elem is vertex_elem:
            vertices = _read_binary_vertices(stream, elem, endian)
        elif elem is face_elem:
            faces = _read_binary_faces(stream, elem, endian)
        else:
            _skip_binary_element(stream, elem, endian)

    assert vertices is not None and faces is not None
    return vertices, faces


def _read_binary_vertices(
    stream: io.BytesIO, elem: _Element, endian: _Endian
) -> np.ndarray:
    dt_fields: list[tuple[str, np.dtype]] = []
    for prop in elem["properties"]:
        if prop[0] != "scalar":
            raise ValueError("List properties in vertex element are not supported")
        _, type_name, prop_name = prop
        dt = np.dtype(_PLY_TYPES[type_name][1]).newbyteorder(endian)
        dt_fields.append((prop_name, dt))

    vertex_dtype = np.dtype(dt_fields)
    raw = stream.read(elem["count"] * vertex_dtype.itemsize)
    data = np.frombuffer(raw, dtype=vertex_dtype)

    return np.column_stack(
        [
            data["x"].astype(np.float64),
            data["y"].astype(np.float64),
            data["z"].astype(np.float64),
        ]
    )


def _read_binary_faces(
    stream: io.BytesIO, elem: _Element, endian: _Endian
) -> np.ndarray:
    # Find the first list property (vertex_indices / vertex_index)
    face_prop = next((p for p in elem["properties"] if p[0] == "list"), None)
    if face_prop is None:
        raise ValueError("Face element has no list property")

    _, count_type, index_type, _prop_name = face_prop
    count_dt = np.dtype(_PLY_TYPES[count_type][1]).newbyteorder(endian)
    index_dt = np.dtype(_PLY_TYPES[index_type][1]).newbyteorder(endian)

    # Fast path: read all face data at once assuming triangular faces
    face_dtype = np.dtype(
        [("count", count_dt), ("i0", index_dt), ("i1", index_dt), ("i2", index_dt)]
    )
    raw = stream.read(elem["count"] * face_dtype.itemsize)
    face_data = np.frombuffer(raw, dtype=face_dtype)

    if not np.all(face_data["count"] == 3):
        raise ValueError(
            "Only triangular faces are supported (every face must have exactly 3 vertices)"
        )

    return np.column_stack(
        [
            face_data["i0"].astype(np.int32),
            face_data["i1"].astype(np.int32),
            face_data["i2"].astype(np.int32),
        ]
    )


def _skip_binary_element(stream: io.BytesIO, elem: _Element, endian: _Endian) -> None:
    """Advance the stream past an element we don't need."""
    all_scalar = all(p[0] == "scalar" for p in elem["properties"])
    if all_scalar:
        stride = sum(np.dtype(_PLY_TYPES[p[1]][1]).itemsize for p in elem["properties"])
        stream.read(elem["count"] * stride)
    else:
        # Row-by-row for elements with list properties
        for _ in range(elem["count"]):
            for prop in elem["properties"]:
                if prop[0] == "scalar":
                    stream.read(np.dtype(_PLY_TYPES[prop[1]][1]).itemsize)
                else:
                    count_size = np.dtype(_PLY_TYPES[prop[1]][1]).itemsize
                    count_fmt = endian + _PLY_TYPES[prop[1]][0]
                    n = struct.unpack(count_fmt, stream.read(count_size))[0]
                    stream.read(n * np.dtype(_PLY_TYPES[prop[2]][1]).itemsize)


# ---------------------------------------------------------------------------
# ASCII reading
# ---------------------------------------------------------------------------


def _load_ascii(
    stream: io.BytesIO,
    elements: list[_Element],
    vertex_elem: _Element,
    face_elem: _Element,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None

    for elem in elements:
        if elem is vertex_elem:
            prop_names = [p[2] for p in elem["properties"] if p[0] == "scalar"]
            x_idx = prop_names.index("x")
            y_idx = prop_names.index("y")
            z_idx = prop_names.index("z")

            vertices = np.empty((elem["count"], 3), dtype=np.float64)
            for i in range(elem["count"]):
                vals = stream.readline().decode("ascii").split()
                vertices[i] = [
                    float(vals[x_idx]),
                    float(vals[y_idx]),
                    float(vals[z_idx]),
                ]

        elif elem is face_elem:
            faces = np.empty((elem["count"], 3), dtype=np.int32)
            for i in range(elem["count"]):
                vals = stream.readline().decode("ascii").split()
                n = int(vals[0])
                if n != 3:
                    raise ValueError(
                        f"Only triangular faces are supported, got face with {n} vertices"
                    )
                faces[i] = [int(vals[1]), int(vals[2]), int(vals[3])]
        else:
            # Skip unknown elements
            for _ in range(elem["count"]):
                stream.readline()

    assert vertices is not None and faces is not None
    return vertices, faces


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_mesh_vf(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Load a PLY triangle mesh from raw bytes.

    Supports ASCII, binary little-endian, and binary big-endian PLY files.

    Args:
        data: Complete PLY file contents as a bytes object.

    Returns:
        ``(vertices, faces)`` where *vertices* is an ``(N, 3)`` float64 array
        and *faces* is an ``(M, 3)`` int32 array of triangle indices.
    """
    stream = io.BytesIO(data)
    fmt, elements = _parse_header(stream)
    vertex_elem, face_elem = _find_elements(elements)

    if fmt == "ascii":
        return _load_ascii(stream, elements, vertex_elem, face_elem)
    elif fmt == "binary_little_endian":
        return _load_binary(stream, elements, vertex_elem, face_elem, "<")
    elif fmt == "binary_big_endian":
        return _load_binary(stream, elements, vertex_elem, face_elem, ">")
    else:
        raise TypeError(f"Unsupported PLY format: {fmt!r}")


def save_mesh_vf(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Save a triangle mesh as binary little-endian PLY bytes.

    Args:
        vertices: ``(N, 3)`` array of vertex positions (any float type).
        faces: ``(M, 3)`` array of triangle vertex indices (any int type).

    Returns:
        Complete PLY file contents as a bytes object.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces must have shape (M, 3) for triangular meshes, got {faces.shape}"
        )

    n_vertices, n_faces = vertices.shape[0], faces.shape[0]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_vertices}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    buf = io.BytesIO()
    buf.write(header.encode("ascii"))

    # Vertices: float32, little-endian
    buf.write(vertices.astype("<f4").tobytes())

    # Faces: packed [uint8 count, int32 i0, int32 i1, int32 i2] per row
    face_dtype = np.dtype([("n", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
    face_data = np.empty(n_faces, dtype=face_dtype)
    face_data["n"] = 3
    face_data["i0"] = faces[:, 0]
    face_data["i1"] = faces[:, 1]
    face_data["i2"] = faces[:, 2]
    buf.write(face_data.tobytes())

    return buf.getvalue()
