"""Procedural mesh generation and MuJoCo mesh extraction."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import numpy.typing as npt


def make_capsule_mesh(
    radius: float,
    half_height: float,
    rings: int = 8,
    segments: int = 16,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Generate a capsule mesh (cylinder with hemispherical caps) along the Z axis.

    The capsule center is at the origin. The cylinder spans from
    ``-half_height`` to ``+half_height`` along Z, with hemisphere caps of the
    given *radius* beyond that.

    Args:
        radius: Capsule radius.
        half_height: Half-length of the cylindrical section.
        rings: Number of latitude rings per hemisphere cap.
        segments: Number of longitude segments around the circumference.

    Returns:
        ``(vertices, faces)`` where vertices is ``(N, 3)`` float32 and
        faces is ``(M, 3)`` int32.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    # Top cap: from pole (+z) down to equator
    # Bottom cap: from equator down to pole (-z)
    for cap in ("top", "bottom"):
        z_sign = 1.0 if cap == "top" else -1.0
        z_offset = z_sign * half_height
        vert_offset = len(verts)

        # Pole vertex
        verts.append([0.0, 0.0, z_offset + z_sign * radius])

        for ring in range(1, rings + 1):
            phi = (math.pi / 2) * ring / rings  # 0 at pole → π/2 at equator
            r = radius * math.sin(phi)
            z = z_offset + z_sign * radius * math.cos(phi)
            for seg in range(segments):
                theta = 2 * math.pi * seg / segments
                verts.append([r * math.cos(theta), r * math.sin(theta), z])

        # Triangles from pole to first ring
        pole = vert_offset
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            a = vert_offset + 1 + seg
            b = vert_offset + 1 + next_seg
            if cap == "top":
                faces.append([pole, a, b])
            else:
                faces.append([pole, b, a])

        # Triangles between rings
        for ring in range(1, rings):
            for seg in range(segments):
                next_seg = (seg + 1) % segments
                curr_base = vert_offset + 1 + (ring - 1) * segments
                next_base = vert_offset + 1 + ring * segments
                a = curr_base + seg
                b = curr_base + next_seg
                c = next_base + seg
                d = next_base + next_seg
                if cap == "top":
                    faces.append([a, c, b])
                    faces.append([b, c, d])
                else:
                    faces.append([a, b, c])
                    faces.append([b, d, c])

    # Cylinder body: connect the equator rings of the two caps.
    # Top cap equator: indices [1 + (rings-1)*segments ... 1 + rings*segments - 1]
    # Bottom cap equator: same offsets but in the bottom cap vertex block.
    top_equator_start = 1 + (rings - 1) * segments
    bottom_cap_offset = 1 + rings * segments  # start of bottom cap vertices
    bottom_equator_start = bottom_cap_offset + 1 + (rings - 1) * segments

    for seg in range(segments):
        next_seg = (seg + 1) % segments
        a = top_equator_start + seg
        b = top_equator_start + next_seg
        c = bottom_equator_start + seg
        d = bottom_equator_start + next_seg
        faces.append([a, c, b])
        faces.append([b, c, d])

    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )


def make_ellipsoid_mesh(
    semi_axes: npt.NDArray[np.floating],
    subdivisions: int = 2,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Generate an ellipsoid mesh by scaling an icosphere.

    Args:
        semi_axes: Array of 3 semi-axis lengths ``(a, b, c)`` for X, Y, Z.
        subdivisions: Number of icosphere subdivision iterations. Higher values
            give smoother results but more triangles.

    Returns:
        ``(vertices, faces)`` where vertices is ``(N, 3)`` float32 and
        faces is ``(M, 3)`` int32.
    """
    verts, faces = _make_icosphere(subdivisions)
    verts = verts * np.array(semi_axes, dtype=np.float32).reshape(1, 3)
    return verts, faces


def extract_mujoco_mesh(
    model: mujoco.MjModel,
    mesh_id: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Extract vertices and faces for a MuJoCo mesh, applying centering transform.

    MuJoCo auto-centers meshes during compilation and stores the offset in
    ``mesh_pos`` and ``mesh_quat``. This function applies that transform so the
    returned vertices are in the geom-local frame.

    Args:
        model: The compiled MuJoCo model.
        mesh_id: Index into the model's mesh arrays.

    Returns:
        ``(vertices, faces)`` where vertices is ``(N, 3)`` float32 and
        faces is ``(M, 3)`` int32.
    """
    vert_adr = model.mesh_vertadr[mesh_id]
    vert_num = model.mesh_vertnum[mesh_id]
    face_adr = model.mesh_faceadr[mesh_id]
    face_num = model.mesh_facenum[mesh_id]

    verts = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
    faces = model.mesh_face[face_adr : face_adr + face_num].copy()

    return verts.astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_icosphere(
    subdivisions: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Generate a unit icosphere via recursive subdivision.

    Returns ``(vertices, faces)`` on the unit sphere.
    """
    # Initial icosahedron vertices
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw_verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    raw_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    verts_arr = np.array(raw_verts, dtype=np.float64)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts_arr, axis=1, keepdims=True)
    verts_arr = verts_arr / norms
    faces_arr = np.array(raw_faces, dtype=np.int32)

    # Subdivide
    for _ in range(subdivisions):
        verts_arr, faces_arr = _subdivide(verts_arr, faces_arr)

    return verts_arr.astype(np.float32), faces_arr


def _subdivide(
    verts: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Subdivide each triangle into 4, projecting new vertices onto unit sphere."""
    edge_midpoints: dict[tuple[int, int], int] = {}

    def get_midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        idx = len(verts_list)
        verts_list.append(mid)
        edge_midpoints[key] = idx
        return idx

    verts_list = list(verts)
    new_faces = []

    for a, b, c in faces:
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])

    return np.array(verts_list, dtype=np.float64), np.array(new_faces, dtype=np.int32)
