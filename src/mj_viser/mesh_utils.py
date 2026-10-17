# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MuJoCo mesh extraction."""

from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt


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


def extract_mujoco_mesh_corners(
    model: mujoco.MjModel,
    mesh_id: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32] | None]:
    """Extract a MuJoCo mesh with one vertex per face corner, and its texture coordinates.

    MuJoCo indexes positions (``mesh_face``) and texture coordinates
    (``mesh_facetexcoord``) separately, so a vertex can carry different texture
    coordinates on different faces. Unsharing the corners keeps both exact.

    Returns:
        ``(vertices, faces, texcoords)``: ``(3F, 3)`` float32 positions, ``(F, 3)``
        int32 faces, and ``(3F, 2)`` float32 texture coordinates in MuJoCo's
        convention (``(0, 0)`` is the first texel row and column), or ``None`` when
        the mesh has none.
    """
    vert_adr = model.mesh_vertadr[mesh_id]
    face_adr = model.mesh_faceadr[mesh_id]
    face_num = model.mesh_facenum[mesh_id]

    corners = model.mesh_face[face_adr : face_adr + face_num].reshape(-1)
    verts = model.mesh_vert[vert_adr + corners].astype(np.float32)
    faces = np.arange(3 * face_num, dtype=np.int32).reshape(-1, 3)

    texcoords = None
    if model.mesh_texcoordnum[mesh_id] > 0:
        tc_adr = model.mesh_texcoordadr[mesh_id]
        tc_corners = model.mesh_facetexcoord[face_adr : face_adr + face_num].reshape(-1)
        texcoords = model.mesh_texcoord[tc_adr + tc_corners].astype(np.float32)
    return verts, faces, texcoords
