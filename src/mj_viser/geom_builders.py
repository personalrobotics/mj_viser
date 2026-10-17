# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Conversion of MuJoCo visual geoms and materials into Viser scene nodes.

One builder per geom type. Each node is created in its geom's frame; the caller
places it at the geom's local pose. The rules follow MuJoCo's own visualizer:

- Color: a material's rgba replaces the geom's only while the geom's rgba is
  MuJoCo's default ``(0.5, 0.5, 0.5, 1)``.
- Textures: 2D textures in a material's ``rgb`` role render on meshes with texture
  coordinates and on planes, and ``(0, 0)`` is the first texel row and column, as
  in MuJoCo and glTF. Like MuJoCo's classic renderer, only 3- and 4-channel
  textures are drawn, a texture's alpha matters only when the geom is transparent,
  and other material roles (``rgba``, ``emissive``, ``normal``, ...) are not drawn.
  Plane coordinates are scaled by the material's ``texrepeat`` (also by the
  half-size when ``texuniform`` is set). Mesh coordinates are scaled by
  ``texrepeat`` only with MuJoCo 3.12 or newer, and before 3.12 plane textures are
  centered on the plane rather than starting at a corner, matching MuJoCo's own
  renderer in each release.
- Planes of size 0 (infinite in MuJoCo) render as a square of half-size
  :data:`INFINITE_PLANE_HALF_SIZE`.
"""

from __future__ import annotations

from typing import Any, Callable

import mujoco
import numpy as np
import numpy.typing as npt
import trimesh
import viser

from mj_viser.mesh_utils import extract_mujoco_mesh, extract_mujoco_mesh_corners

DEFAULT_GEOM_RGBA = np.array([0.5, 0.5, 0.5, 1.0])
INFINITE_PLANE_HALF_SIZE = 50.0

_RGB_ROLE = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
# MuJoCo 3.12.0 changed its renderer's texture layout: before it, texrepeat is ignored
# on mesh texture coordinates and plane textures are centered on the plane; from it,
# mesh coordinates are scaled by texrepeat and plane textures start at a corner.
_MUJOCO_3_12 = tuple(int(x) for x in mujoco.__version__.split(".")[:2]) >= (3, 12)
_TEXTURED_TYPES = {int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_PLANE)}


def geom_rgba(model: mujoco.MjModel, geom_id: int) -> npt.NDArray[np.float64]:
    """The rgba MuJoCo draws a geom with, after material resolution."""
    mat_id = model.geom_matid[geom_id]
    rgba = model.geom_rgba[geom_id]
    # Exact comparison, as MuJoCo does: a near-default explicit color is still explicit.
    if mat_id >= 0 and np.array_equal(rgba, DEFAULT_GEOM_RGBA):
        rgba = model.mat_rgba[mat_id]
    return np.asarray(rgba, dtype=np.float64)


def _texture_id(model: mujoco.MjModel, geom_id: int) -> int:
    mat_id = model.geom_matid[geom_id]
    return int(model.mat_texid[mat_id, _RGB_ROLE]) if mat_id >= 0 else -1


def unrendered_texture_reason(model: mujoco.MjModel, geom_id: int) -> str | None:
    """Why a geom's texture is not drawn (its base color is used instead), or None."""
    tex_id = _texture_id(model, geom_id)
    if tex_id < 0:
        return None
    geom_type = int(model.geom_type[geom_id])
    if model.tex_type[tex_id] != mujoco.mjtTexture.mjTEXTURE_2D:
        return "only 2D textures are rendered; base color used"
    if (channels := int(model.tex_nchannel[tex_id])) not in (3, 4):
        return f"{channels}-channel textures are not rendered (MuJoCo's classic renderer skips them); base color used"
    if geom_type not in _TEXTURED_TYPES:
        type_name = mujoco.mjtGeom(geom_type).name.removeprefix("mjGEOM_").lower()
        return f"textures on {type_name} geoms are not rendered; base color used"
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH and model.mesh_texcoordnum[model.geom_dataid[geom_id]] == 0:
        return "mesh has no texture coordinates; base color used"
    return None


def _renders_texture(model: mujoco.MjModel, geom_id: int) -> bool:
    return _texture_id(model, geom_id) >= 0 and unrendered_texture_reason(model, geom_id) is None


def _color(rgba: npt.NDArray[np.float64]) -> tuple[int, int, int]:
    return (round(rgba[0] * 255), round(rgba[1] * 255), round(rgba[2] * 255))


def _opacity(rgba: npt.NDArray[np.float64]) -> float | None:
    return float(rgba[3]) if rgba[3] < 1.0 else None


def _simple_mesh(
    scene: viser.SceneApi,
    name: str,
    model: mujoco.MjModel,
    geom_id: int,
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    side: str = "front",
) -> viser.MeshHandle:
    rgba = geom_rgba(model, geom_id)
    return scene.add_mesh_simple(
        name,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.uint32),
        color=_color(rgba),
        opacity=_opacity(rgba),
        flat_shading=False,
        side=side,  # type: ignore[arg-type]
    )


def _textured_mesh(
    scene: viser.SceneApi,
    name: str,
    model: mujoco.MjModel,
    geom_id: int,
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    texcoords: npt.NDArray[np.floating],
) -> viser.GlbHandle:
    """A mesh with its geom's 2D texture, exported as glTF.

    ``texcoords`` use MuJoCo's convention, which matches glTF's. trimesh stores
    texture coordinates with v flipped and flips them back on export, so they are
    flipped here to arrive unchanged in the GLB.
    """
    from PIL import Image

    tex_id = _texture_id(model, geom_id)
    height, width, channels = model.tex_height[tex_id], model.tex_width[tex_id], model.tex_nchannel[tex_id]
    adr = model.tex_adr[tex_id]
    pixels = model.tex_data[adr : adr + height * width * channels].reshape(height, width, channels)
    rgba = geom_rgba(model, geom_id)
    material = trimesh.visual.material.PBRMaterial(  # type: ignore[no-untyped-call]
        baseColorFactor=[float(c) for c in rgba],
        baseColorTexture=Image.fromarray(np.ascontiguousarray(pixels)),
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode="BLEND" if rgba[3] < 1.0 else "OPAQUE",
    )
    uv = np.column_stack([texcoords[:, 0], 1.0 - texcoords[:, 1]])
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)  # type: ignore[no-untyped-call]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return scene.add_mesh_trimesh(name, mesh=mesh)


def build_box(scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel) -> viser.BoxHandle:
    size = model.geom_size[geom_id]
    rgba = geom_rgba(model, geom_id)
    return scene.add_box(
        name,
        dimensions=(float(size[0] * 2), float(size[1] * 2), float(size[2] * 2)),
        color=_color(rgba),
        opacity=_opacity(rgba),
    )


def build_sphere(scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel) -> viser.IcosphereHandle:
    rgba = geom_rgba(model, geom_id)
    return scene.add_icosphere(
        name,
        radius=float(model.geom_size[geom_id][0]),
        color=_color(rgba),
        opacity=_opacity(rgba),
    )


def build_cylinder(scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel) -> viser.CylinderHandle:
    size = model.geom_size[geom_id]
    rgba = geom_rgba(model, geom_id)
    return scene.add_cylinder(
        name,
        radius=float(size[0]),
        height=float(size[1] * 2),
        color=_color(rgba),
        opacity=_opacity(rgba),
    )


def build_capsule(scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel) -> viser.MeshHandle:
    radius, half_length = model.geom_size[geom_id][:2]
    capsule = trimesh.creation.capsule(height=float(2 * half_length), radius=float(radius), count=[24, 24])
    return _simple_mesh(scene, name, model, geom_id, capsule.vertices, capsule.faces)


def build_ellipsoid(scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel) -> viser.MeshHandle:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    return _simple_mesh(scene, name, model, geom_id, sphere.vertices * model.geom_size[geom_id], sphere.faces)


def build_plane(
    scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel
) -> viser.MeshHandle | viser.GlbHandle:
    half_x, half_y = (float(s) if s > 0 else INFINITE_PLANE_HALF_SIZE for s in model.geom_size[geom_id][:2])
    vertices = np.array([[-half_x, -half_y, 0], [half_x, -half_y, 0], [half_x, half_y, 0], [-half_x, half_y, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    if not _renders_texture(model, geom_id):
        return _simple_mesh(scene, name, model, geom_id, vertices, faces)
    mat_id = model.geom_matid[geom_id]
    repeat = model.mat_texrepeat[mat_id] * (np.array([half_x, half_y]) if model.mat_texuniform[mat_id] else 1.0)
    across = np.column_stack([vertices[:, 0] / (2 * half_x), -vertices[:, 1] / (2 * half_y)])  # in [-0.5, 0.5]
    texcoords = (across + 0.5) * repeat if _MUJOCO_3_12 else across * repeat + 0.5
    return _textured_mesh(scene, name, model, geom_id, vertices, faces, texcoords)


def build_mesh(
    scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel
) -> viser.MeshHandle | viser.GlbHandle:
    mesh_id = model.geom_dataid[geom_id]
    if _renders_texture(model, geom_id):
        vertices, faces, texcoords = extract_mujoco_mesh_corners(model, mesh_id)
        assert texcoords is not None
        if _MUJOCO_3_12:
            texcoords = texcoords * model.mat_texrepeat[model.geom_matid[geom_id]]
        return _textured_mesh(scene, name, model, geom_id, vertices, faces, texcoords)
    vertices, faces = extract_mujoco_mesh(model, mesh_id)
    # Both sides: STL meshes may have inverted normals.
    return _simple_mesh(scene, name, model, geom_id, vertices, faces, side="double")


GEOM_BUILDERS: dict[int, Callable[[viser.SceneApi, str, int, mujoco.MjModel], Any]] = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): build_plane,
    int(mujoco.mjtGeom.mjGEOM_SPHERE): build_sphere,
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): build_capsule,
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): build_ellipsoid,
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): build_cylinder,
    int(mujoco.mjtGeom.mjGEOM_BOX): build_box,
    int(mujoco.mjtGeom.mjGEOM_MESH): build_mesh,
}
