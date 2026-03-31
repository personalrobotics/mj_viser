"""Per-geom-type builder functions that create Viser scene nodes from MuJoCo geoms."""

from __future__ import annotations

from typing import Any, Callable

import mujoco
import numpy as np
import viser

from mj_viser.mesh_utils import (
    extract_mujoco_mesh,
    extract_mujoco_mesh_textured,
    make_capsule_mesh,
    make_ellipsoid_mesh,
)


def _resolve_color(model: mujoco.MjModel, geom_id: int) -> tuple[int, int, int]:
    """Get RGB color (0-255) for a geom, respecting material overrides."""
    mat_id = model.geom_matid[geom_id]
    if mat_id >= 0:
        rgba = model.mat_rgba[mat_id]
    else:
        rgba = model.geom_rgba[geom_id]
    # Default gray if color is all zeros
    if rgba[:3].sum() == 0 and rgba[3] == 0:
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
    return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


def _resolve_opacity(model: mujoco.MjModel, geom_id: int) -> float | None:
    """Return opacity if < 1.0, else None (fully opaque)."""
    mat_id = model.geom_matid[geom_id]
    alpha = model.mat_rgba[mat_id][3] if mat_id >= 0 else model.geom_rgba[geom_id][3]
    return float(alpha) if alpha < 1.0 else None


def _geom_name(geom_id: int) -> str:
    return f"/mujoco/geoms/geom_{geom_id}"


def build_box(
    scene: viser.SceneApi, geom_id: int, model: mujoco.MjModel
) -> viser.BoxHandle:
    size = model.geom_size[geom_id]
    return scene.add_box(
        _geom_name(geom_id),
        dimensions=(float(size[0] * 2), float(size[1] * 2), float(size[2] * 2)),
        color=_resolve_color(model, geom_id),
        opacity=_resolve_opacity(model, geom_id),
    )


def build_sphere(
    scene: viser.SceneApi, geom_id: int, model: mujoco.MjModel
) -> viser.IcosphereHandle:
    radius = float(model.geom_size[geom_id][0])
    return scene.add_icosphere(
        _geom_name(geom_id),
        radius=radius,
        color=_resolve_color(model, geom_id),
        opacity=_resolve_opacity(model, geom_id),
    )


def build_cylinder(
    scene: viser.SceneApi, geom_id: int, model: mujoco.MjModel
) -> viser.CylinderHandle:
    size = model.geom_size[geom_id]
    return scene.add_cylinder(
        _geom_name(geom_id),
        radius=float(size[0]),
        height=float(size[1] * 2),
        color=_resolve_color(model, geom_id),
        opacity=_resolve_opacity(model, geom_id),
    )


def build_capsule(
    scene: viser.SceneApi, geom_id: int, model: mujoco.MjModel
) -> viser.MeshHandle:
    size = model.geom_size[geom_id]
    radius = float(size[0])
    half_height = float(size[1])
    verts, faces = make_capsule_mesh(radius, half_height)
    return scene.add_mesh_simple(
        _geom_name(geom_id),
        vertices=verts,
        faces=faces,
        color=_resolve_color(model, geom_id),
        opacity=_resolve_opacity(model, geom_id),
        flat_shading=False,
    )


def build_ellipsoid(
    scene: viser.SceneApi, geom_id: int, model: mujoco.MjModel
) -> viser.MeshHandle:
    semi_axes = model.geom_size[geom_id]
    verts, faces = make_ellipsoid_mesh(semi_axes)
    return scene.add_mesh_simple(
        _geom_name(geom_id),
        vertices=verts,
        faces=faces,
        color=_resolve_color(model, geom_id),
        opacity=_resolve_opacity(model, geom_id),
        flat_shading=False,
    )


def build_mesh(
    scene: viser.SceneApi, geom_id: int, model: mujoco.MjModel
) -> viser.MeshHandle:
    mesh_id = model.geom_dataid[geom_id]
    verts, faces, texcoords, texture_rgb = extract_mujoco_mesh_textured(
        model, mesh_id, geom_id,
    )

    # Use textured mesh if UV + texture data available
    if texcoords is not None and texture_rgb is not None and len(texcoords) == len(verts):
        import trimesh
        from PIL import Image

        image = Image.fromarray(texture_rgb)
        material = trimesh.visual.material.SimpleMaterial(image=image)
        visual = trimesh.visual.TextureVisuals(uv=texcoords, material=material)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual)
        return scene.add_mesh_trimesh(
            _geom_name(geom_id),
            mesh=mesh,
        )

    # Fallback: solid color
    return scene.add_mesh_simple(
        _geom_name(geom_id),
        vertices=verts,
        faces=faces,
        color=_resolve_color(model, geom_id),
        opacity=_resolve_opacity(model, geom_id),
        flat_shading=False,
    )


# Dispatch table: MuJoCo geom type → builder function.
GEOM_BUILDERS: dict[int, Callable[[viser.SceneApi, int, mujoco.MjModel], Any]] = {
    mujoco.mjtGeom.mjGEOM_BOX: build_box,
    mujoco.mjtGeom.mjGEOM_SPHERE: build_sphere,
    mujoco.mjtGeom.mjGEOM_CYLINDER: build_cylinder,
    mujoco.mjtGeom.mjGEOM_CAPSULE: build_capsule,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: build_ellipsoid,
    mujoco.mjtGeom.mjGEOM_MESH: build_mesh,
}
