# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import io
import struct

import numpy as np
import pytest
from alpasim_physics.ply_io import load_mesh_vf, save_mesh_vf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRIANGLE_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
TRIANGLE_FACES = np.array([[0, 1, 2]], dtype=np.int32)

QUAD_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
QUAD_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _make_ascii_ply(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Build a minimal ASCII PLY from arrays."""
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v in vertices:
        lines.append(f"{v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"3 {f[0]} {f[1]} {f[2]}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _make_binary_ply(
    vertices: np.ndarray, faces: np.ndarray, endian: str = "<"
) -> bytes:
    """Build a minimal binary PLY from arrays.

    *endian* is ``"<"`` for little-endian or ``">"`` for big-endian.
    """
    fmt_name = "binary_little_endian" if endian == "<" else "binary_big_endian"
    header = (
        "ply\n"
        f"format {fmt_name} 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    buf = io.BytesIO()
    buf.write(header.encode("ascii"))

    float_fmt = endian + "f"
    for v in vertices:
        buf.write(struct.pack(float_fmt, v[0]))
        buf.write(struct.pack(float_fmt, v[1]))
        buf.write(struct.pack(float_fmt, v[2]))

    int_fmt = endian + "i"
    for f in faces:
        buf.write(struct.pack("B", 3))
        buf.write(struct.pack(int_fmt, f[0]))
        buf.write(struct.pack(int_fmt, f[1]))
        buf.write(struct.pack(int_fmt, f[2]))

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests — loading
# ---------------------------------------------------------------------------


class TestLoadAscii:
    def test_single_triangle(self) -> None:
        data = _make_ascii_ply(TRIANGLE_VERTICES, TRIANGLE_FACES)
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, TRIANGLE_VERTICES)
        np.testing.assert_array_equal(faces, TRIANGLE_FACES)

    def test_two_triangles(self) -> None:
        data = _make_ascii_ply(QUAD_VERTICES, QUAD_FACES)
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, QUAD_VERTICES)
        np.testing.assert_array_equal(faces, QUAD_FACES)


class TestLoadBinaryLE:
    def test_single_triangle(self) -> None:
        data = _make_binary_ply(TRIANGLE_VERTICES, TRIANGLE_FACES, "<")
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, TRIANGLE_VERTICES, atol=1e-6)
        np.testing.assert_array_equal(faces, TRIANGLE_FACES)

    def test_two_triangles(self) -> None:
        data = _make_binary_ply(QUAD_VERTICES, QUAD_FACES, "<")
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, QUAD_VERTICES, atol=1e-6)
        np.testing.assert_array_equal(faces, QUAD_FACES)


class TestLoadBinaryBE:
    def test_single_triangle(self) -> None:
        data = _make_binary_ply(TRIANGLE_VERTICES, TRIANGLE_FACES, ">")
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, TRIANGLE_VERTICES, atol=1e-6)
        np.testing.assert_array_equal(faces, TRIANGLE_FACES)

    def test_two_triangles(self) -> None:
        data = _make_binary_ply(QUAD_VERTICES, QUAD_FACES, ">")
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, QUAD_VERTICES, atol=1e-6)
        np.testing.assert_array_equal(faces, QUAD_FACES)


class TestLoadExtraProperties:
    """Verify that extra vertex properties (normals, colors, …) are skipped."""

    def test_ascii_with_normals(self) -> None:
        lines = [
            "ply",
            "format ascii 1.0",
            "element vertex 3",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "element face 1",
            "property list uchar int vertex_indices",
            "end_header",
            "0.0 0.0 0.0  0.0 0.0 1.0",
            "1.0 0.0 0.0  0.0 0.0 1.0",
            "0.0 1.0 0.0  0.0 0.0 1.0",
            "3 0 1 2",
        ]
        data = ("\n".join(lines) + "\n").encode("ascii")
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, TRIANGLE_VERTICES)
        np.testing.assert_array_equal(faces, TRIANGLE_FACES)

    def test_binary_le_with_normals(self) -> None:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        buf = io.BytesIO()
        buf.write(header.encode("ascii"))
        for v in TRIANGLE_VERTICES:
            buf.write(struct.pack("<fff", v[0], v[1], v[2]))
            buf.write(struct.pack("<fff", 0.0, 0.0, 1.0))  # normals
        buf.write(struct.pack("B", 3))
        buf.write(struct.pack("<iii", 0, 1, 2))

        verts, faces = load_mesh_vf(buf.getvalue())
        np.testing.assert_allclose(verts, TRIANGLE_VERTICES, atol=1e-6)
        np.testing.assert_array_equal(faces, TRIANGLE_FACES)


# ---------------------------------------------------------------------------
# Tests — round-trip (save then load)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_single_triangle(self) -> None:
        data = save_mesh_vf(TRIANGLE_VERTICES, TRIANGLE_FACES)
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, TRIANGLE_VERTICES, atol=1e-6)
        np.testing.assert_array_equal(faces, TRIANGLE_FACES)

    def test_two_triangles(self) -> None:
        data = save_mesh_vf(QUAD_VERTICES, QUAD_FACES)
        verts, faces = load_mesh_vf(data)
        np.testing.assert_allclose(verts, QUAD_VERTICES, atol=1e-6)
        np.testing.assert_array_equal(faces, QUAD_FACES)

    def test_large_grid(self) -> None:
        """Round-trip a mesh similar to what the physics tests generate."""
        n = 20
        x = np.linspace(-5, 5, n)
        y = np.linspace(-5, 5, n)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                faces.append([idx, idx + 1, idx + n])
                faces.append([idx + 1, idx + n + 1, idx + n])
        faces = np.array(faces, dtype=np.int32)

        data = save_mesh_vf(vertices, faces)
        v_out, f_out = load_mesh_vf(data)

        np.testing.assert_allclose(v_out, vertices, atol=1e-6)
        np.testing.assert_array_equal(f_out, faces)


# ---------------------------------------------------------------------------
# Tests — saving
# ---------------------------------------------------------------------------


class TestSave:
    def test_output_starts_with_ply_header(self) -> None:
        data = save_mesh_vf(TRIANGLE_VERTICES, TRIANGLE_FACES)
        assert data.startswith(b"ply\n")

    def test_binary_le_format(self) -> None:
        data = save_mesh_vf(TRIANGLE_VERTICES, TRIANGLE_FACES)
        assert b"binary_little_endian" in data


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_not_a_ply(self) -> None:
        with pytest.raises(ValueError, match="Not a PLY file"):
            load_mesh_vf(b"NOT A PLY FILE")

    def test_missing_vertex_element(self) -> None:
        header = (
            "ply\n"
            "format ascii 1.0\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "3 0 1 2\n"
        )
        with pytest.raises(ValueError, match="no 'vertex' element"):
            load_mesh_vf(header.encode("ascii"))

    def test_missing_face_element(self) -> None:
        header = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            "0 0 0\n1 0 0\n0 1 0\n"
        )
        with pytest.raises(ValueError, match="no 'face' element"):
            load_mesh_vf(header.encode("ascii"))

    def test_non_triangular_faces_ascii(self) -> None:
        lines = [
            "ply",
            "format ascii 1.0",
            "element vertex 4",
            "property float x",
            "property float y",
            "property float z",
            "element face 1",
            "property list uchar int vertex_indices",
            "end_header",
            "0 0 0",
            "1 0 0",
            "1 1 0",
            "0 1 0",
            "4 0 1 2 3",  # quad face
        ]
        with pytest.raises(ValueError, match="triangular"):
            load_mesh_vf(("\n".join(lines) + "\n").encode("ascii"))

    def test_save_non_triangular_faces(self) -> None:
        quad_faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
        with pytest.raises(ValueError, match="triangular"):
            save_mesh_vf(TRIANGLE_VERTICES, quad_faces)

    def test_save_bad_vertex_shape(self) -> None:
        bad_verts = np.array([[0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="vertices must have shape"):
            save_mesh_vf(bad_verts, TRIANGLE_FACES)

    def test_unknown_format(self) -> None:
        header = (
            "ply\n"
            "format unknown_format 1.0\n"
            "element vertex 0\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face 0\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        with pytest.raises(ValueError, match="Unsupported PLY format"):
            load_mesh_vf(header.encode("ascii"))


# ---------------------------------------------------------------------------
# Tests — loading the existing binary test fixture
# ---------------------------------------------------------------------------


class TestExistingFixture:
    """Verify we can load the binary LE PLY fixture used by test_backend."""

    def test_load_mesh_ground_ply(self) -> None:
        data = open("tests/data/mesh_ground.ply", "rb").read()
        verts, faces = load_mesh_vf(data)
        assert verts.shape == (1688260, 3)
        assert faces.shape == (3376506, 3)
        assert verts.dtype == np.float64
        assert faces.dtype == np.int32
