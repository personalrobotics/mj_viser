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
  Mesh and plane coordinates are scaled by the material's ``texrepeat`` (on planes,
  also by the half-size when ``texuniform`` is set), and plane tiling starts at a
  corner, as in MuJoCo 3.12 and later.
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
_TEXTURED_TYPES = {int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_PLANE)}


def _resolved_rgba(model: mujoco.MjModel, rgba: npt.NDArray[np.floating], mat_id: int) -> npt.NDArray[np.float64]:
    # Exact comparison, as MuJoCo does: a near-default explicit color is still explicit.
    if mat_id >= 0 and np.array_equal(rgba, DEFAULT_GEOM_RGBA):
        rgba = model.mat_rgba[mat_id]
    return np.asarray(rgba, dtype=np.float64)


def geom_rgba(model: mujoco.MjModel, geom_id: int) -> npt.NDArray[np.float64]:
    """The rgba MuJoCo draws a geom with, after material resolution."""
    return _resolved_rgba(model, model.geom_rgba[geom_id], int(model.geom_matid[geom_id]))


def site_rgba(model: mujoco.MjModel, site_id: int) -> npt.NDArray[np.float64]:
    """The rgba MuJoCo draws a site with; sites resolve materials like geoms."""
    return _resolved_rgba(model, model.site_rgba[site_id], int(model.site_matid[site_id]))


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
    rgba: npt.NDArray[np.float64],
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    side: str = "front",
) -> viser.MeshHandle:
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


PRIMITIVE_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_BOX),
}


def build_primitive(
    scene: viser.SceneApi, name: str, geom_type: int, size: npt.NDArray[np.floating], rgba: npt.NDArray[np.float64]
) -> Any:
    """A MuJoCo primitive shape (geom or site) from its type, size parameters, and rgba."""
    color, opacity = _color(rgba), _opacity(rgba)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        dims = (float(size[0] * 2), float(size[1] * 2), float(size[2] * 2))
        return scene.add_box(name, dimensions=dims, color=color, opacity=opacity)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return scene.add_icosphere(name, radius=float(size[0]), color=color, opacity=opacity)
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return scene.add_cylinder(name, radius=float(size[0]), height=float(size[1] * 2), color=color, opacity=opacity)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        capsule = trimesh.creation.capsule(height=float(2 * size[1]), radius=float(size[0]), count=[24, 24])
        return _simple_mesh(scene, name, rgba, capsule.vertices, capsule.faces)
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        return _simple_mesh(scene, name, rgba, sphere.vertices * np.asarray(size[:3]), sphere.faces)
    raise ValueError(f"not a primitive geom type: {geom_type}")


def _build_geom_primitive(scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel) -> Any:
    return build_primitive(
        scene, name, int(model.geom_type[geom_id]), model.geom_size[geom_id], geom_rgba(model, geom_id)
    )


def build_plane(
    scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel
) -> viser.MeshHandle | viser.GlbHandle:
    half_x, half_y = (float(s) if s > 0 else INFINITE_PLANE_HALF_SIZE for s in model.geom_size[geom_id][:2])
    vertices = np.array([[-half_x, -half_y, 0], [half_x, -half_y, 0], [half_x, half_y, 0], [-half_x, half_y, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    if not _renders_texture(model, geom_id):
        return _simple_mesh(scene, name, geom_rgba(model, geom_id), vertices, faces)
    mat_id = model.geom_matid[geom_id]
    repeat = model.mat_texrepeat[mat_id] * (np.array([half_x, half_y]) if model.mat_texuniform[mat_id] else 1.0)
    across = np.column_stack([vertices[:, 0] / (2 * half_x), -vertices[:, 1] / (2 * half_y)])  # in [-0.5, 0.5]
    texcoords = (across + 0.5) * repeat
    return _textured_mesh(scene, name, model, geom_id, vertices, faces, texcoords)


def build_mesh(
    scene: viser.SceneApi, name: str, geom_id: int, model: mujoco.MjModel
) -> viser.MeshHandle | viser.GlbHandle:
    mesh_id = model.geom_dataid[geom_id]
    if _renders_texture(model, geom_id):
        vertices, faces, texcoords = extract_mujoco_mesh_corners(model, mesh_id)
        assert texcoords is not None
        repeat = model.mat_texrepeat[model.geom_matid[geom_id]]
        return _textured_mesh(scene, name, model, geom_id, vertices, faces, texcoords * repeat)
    vertices, faces = extract_mujoco_mesh(model, mesh_id)
    # Both sides: STL meshes may have inverted normals.
    return _simple_mesh(scene, name, geom_rgba(model, geom_id), vertices, faces, side="double")


GEOM_BUILDERS: dict[int, Callable[[viser.SceneApi, str, int, mujoco.MjModel], Any]] = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): build_plane,
    **{t: _build_geom_primitive for t in PRIMITIVE_TYPES},
    int(mujoco.mjtGeom.mjGEOM_MESH): build_mesh,
}
