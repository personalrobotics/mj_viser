# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

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
    """Extract vertices and faces for a MuJoCo mesh.

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


def extract_mujoco_mesh_textured(
    model: mujoco.MjModel,
    mesh_id: int,
    geom_id: int,
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32] | None, npt.NDArray[np.uint8] | None
]:
    """Extract vertices, faces, UV coordinates, and texture for a MuJoCo mesh.

    Args:
        model: The compiled MuJoCo model.
        mesh_id: Index into the model's mesh arrays.
        geom_id: Geom index (to resolve material → texture).

    Returns:
        ``(vertices, faces, texcoords, texture_rgb)`` where:
        - vertices: ``(N, 3)`` float32
        - faces: ``(M, 3)`` int32
        - texcoords: ``(N, 2)`` float32 or None if no UV data
        - texture_rgb: ``(H, W, 3)`` uint8 or None if no texture
    """
    verts, faces = extract_mujoco_mesh(model, mesh_id)

    # Extract UV coordinates
    texcoords = None
    tc_num = model.mesh_texcoordnum[mesh_id]
    if tc_num > 0:
        tc_adr = model.mesh_texcoordadr[mesh_id]
        raw_tc = model.mesh_texcoord[tc_adr : tc_adr + tc_num].copy().astype(np.float32)

        if tc_num == len(verts):
            # Per-vertex UVs — use directly
            texcoords = raw_tc
        else:
            # Per-face-vertex UVs stored as quad corners: 4 texcoords per pair
            # of triangulated faces. Unpack by duplicating vertices.
            face_num = model.mesh_facenum[mesh_id]
            n_quads = face_num // 2

            if tc_num == n_quads * 4:
                # Quad-based UV layout: each quad has 4 texcoords
                # Face 2q uses tc [4q, 4q+1, 4q+2], face 2q+1 uses [4q, 4q+2, 4q+3]
                tc_indices = []
                for q in range(n_quads):
                    base = q * 4
                    tc_indices.extend([base, base + 1, base + 2])  # tri 0
                    tc_indices.extend([base, base + 2, base + 3])  # tri 1
                # Handle any remaining odd triangle
                if face_num % 2 == 1:
                    base = n_quads * 4
                    tc_indices.extend([base, base + 1, base + 2])

                tc_per_face_vert = raw_tc[tc_indices]  # (F*3, 2)
                new_verts = verts[faces.flatten()]  # (F*3, 3)
                texcoords = tc_per_face_vert.astype(np.float32)
                faces = np.arange(face_num * 3, dtype=np.int32).reshape(-1, 3)
                verts = new_verts.astype(np.float32)
            else:
                # Unknown layout — skip texturing
                texcoords = None

    # Flip V coordinate: MuJoCo uses top-left origin, OpenGL uses bottom-left
    if texcoords is not None:
        texcoords = texcoords.copy()
        texcoords[:, 1] = 1.0 - texcoords[:, 1]

    # Extract texture image
    texture_rgb = None
    mat_id = model.geom_matid[geom_id]
    if mat_id >= 0:
        tex_id = model.mat_texid[mat_id][1]  # index 1 = "2d" texture type
        if tex_id >= 0:
            h = model.tex_height[tex_id]
            w = model.tex_width[tex_id]
            nc = model.tex_nchannel[tex_id]
            adr = model.tex_adr[tex_id]
            tex_data = model.tex_data[adr : adr + h * w * nc].reshape(h, w, nc)
            if nc == 3:
                texture_rgb = tex_data.copy()
            elif nc == 4:
                texture_rgb = tex_data[:, :, :3].copy()

    return verts, faces, texcoords, texture_rgb


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
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    raw_faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
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
