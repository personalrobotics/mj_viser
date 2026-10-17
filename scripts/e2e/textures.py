# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Texture orientation, scale, and channel handling against MuJoCo's renderer."""

from __future__ import annotations

import io
import os
from typing import Any

import mujoco
import numpy as np
import PIL.Image

import mj_viser
from e2e.oracles import glb_color_at, glb_primitive

QUADRANTS_PNG_SIZE = 64


TEXTURE_MJCF = """
<mujoco>
  <asset>
    <texture name="q" type="2d" file="quadrants.png"/>
    <material name="slab_m" texture="q" emission="1" texrepeat="2 2"/>
    <material name="plane_m" texture="q" emission="1" texrepeat="2 2"/>
    <material name="uniform_m" texture="q" emission="1" texrepeat="1 1" texuniform="true"/>
    <texture name="qa" type="2d" file="quadrants_alpha.png" nchannel="4"/>
    <material name="rgba_m" texture="qa" emission="1"/>
    <mesh name="slab" vertex="-.5 -.5 .01  .5 -.5 .01  .5 .5 .01  -.5 .5 .01  -.5 -.5 0  .5 -.5 0  .5 .5 0  -.5 .5 0"
          texcoord="0 0  1 0  1 1  0 1  0 0  1 0  1 1  0 1"
          face="0 1 2  0 2 3  4 6 5  4 7 6  0 4 5  0 5 1  1 5 6  1 6 2  2 6 7  2 7 3  3 7 4  3 4 0"/>
  </asset>
  <visual>
    <global offwidth="1200" offheight="1200"/>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
  </visual>
  <worldbody>
    <geom name="slab" type="mesh" mesh="slab" material="slab_m" pos="-1.2 0 0"/>
    <geom name="plane" type="plane" size=".5 .5 .1" material="plane_m"/>
    <geom name="uniform" type="plane" size="1 .5 .1" material="uniform_m" pos="1.7 0 0"/>
    <!-- 4-channel texture on an opaque material: its alpha is ignored, as in MuJoCo. -->
    <geom name="rgba" type="plane" size=".5 .5 .1" material="rgba_m" pos="0 1.3 0"/>
    <camera name="top" pos="0 0 40" fovy="10"/>
  </worldbody>
</mujoco>
"""


def texture_scenario() -> dict[str, Any]:
    """Texture orientation and scale against MuJoCo's own renderer.

    A four-color texture (red, yellow / blue, cyan) on a mesh with texrepeat 2, a
    plane with texrepeat 2, and a texuniform plane is rendered top-down by MuJoCo
    (emission 1, ambient-only light, so pixels are texture colors). At 64 points
    on each surface, the color glTF assigns in mj_viser's GLB (texture coordinate
    interpolated over the containing triangle, origin top-left, REPEAT wrap) must
    match MuJoCo's pixel.
    """
    q = np.zeros((QUADRANTS_PNG_SIZE, QUADRANTS_PNG_SIZE, 3), np.uint8)
    half = QUADRANTS_PNG_SIZE // 2
    q[:half, :, 0] = 255
    q[half:, :, 2] = 255
    q[:, half:, 1] = 255
    png = io.BytesIO()
    PIL.Image.fromarray(q).save(png, format="PNG")
    qa = np.dstack([q, np.full(q.shape[:2], 255, np.uint8)])
    qa[half:, :, 3] = 64  # bottom half mostly transparent
    png_alpha = io.BytesIO()
    PIL.Image.fromarray(qa, "RGBA").save(png_alpha, format="PNG")
    model = mujoco.MjModel.from_xml_string(
        TEXTURE_MJCF, assets={"quadrants.png": png.getvalue(), "quadrants_alpha.png": png_alpha.getvalue()}
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, 1200, 1200)
    except Exception as e:  # no OpenGL backend on this machine
        if os.environ.get("MJ_VISER_REQUIRE_RENDER"):
            raise
        return {"skipped": f"MuJoCo renderer unavailable: {type(e).__name__}"}
    renderer.update_scene(data, camera="top")
    shot = renderer.render()
    renderer.close()
    scale = 1200 / (2 * 40 * np.tan(np.radians(5)))  # pixels per meter on the z=0 plane

    samples = 0
    with mj_viser.MujocoViewer(model, port=0) as viewer:
        handles = viewer._scene_mgr._geom_handles  # type: ignore[union-attr]
        for name in ("slab", "plane", "uniform", "rgba"):
            g = model.geom(name).id
            glb = glb_primitive(handles[g].glb_data)
            half_xy = model.geom_size[g][:2] if name != "slab" else np.array([0.5, 0.5])
            rot = data.geom_xmat[g].reshape(3, 3)
            for fx in np.linspace(-0.875, 0.875, 8):
                for fy in np.linspace(-0.875, 0.875, 8):
                    world = data.geom_xpos[g] + np.array([fx * half_xy[0], fy * half_xy[1], 0.0])
                    if name == "slab":
                        world[2] = 0.01  # top face
                    truth = shot[int(600 - world[1] * scale), int(600 + world[0] * scale)].astype(float)
                    got = glb_color_at(glb, rot.T @ (world - data.geom_xpos[g]))
                    if np.abs(truth - got).max() > 2:
                        raise SystemExit(f"texture on {name} at {world.round(3)}: MuJoCo {truth}, GLB {got}")
                    samples += 1
    return {
        "samples_matching_mujoco_renderer": samples,
        "surfaces": [
            "mesh texrepeat 2",
            "plane texrepeat 2",
            "plane texuniform",
            "opaque plane, 4-channel texture",
        ],
        "single_channel": single_channel_texture(q[..., 0]),
    }


GRAY_MJCF = """
<mujoco>
  <asset>
    <!-- From a file: before 3.15, MuJoCo corrupts memory compiling a builtin nchannel 1 texture (3.15 rejects it). -->
    <texture name="gray" type="2d" file="gray.png" nchannel="1"/>
    <material name="gray_m" texture="gray" emission="1"/>
  </asset>
  <visual><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
  <worldbody>
    <geom name="gray" type="plane" size=".5 .5 .1" material="gray_m"/>
    <camera name="top" pos="0 0 2" fovy="40"/>
  </worldbody>
</mujoco>
"""


def single_channel_texture(gray: np.ndarray) -> dict[str, Any]:
    """A valid 1-channel texture opens, renders in the base color, and is reported.

    MuJoCo's classic renderer does not draw 1-channel textures: it skips the texture,
    so the base color shows.
    """
    buf = io.BytesIO()
    PIL.Image.fromarray(gray, "L").save(buf, format="PNG")
    model = mujoco.MjModel.from_xml_string(GRAY_MJCF, assets={"gray.png": buf.getvalue()})
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mj_viser.MujocoViewer(model, port=0) as viewer:
        node = viewer._scene_mgr._geom_handles[0]  # type: ignore[union-attr]
        if type(node).__name__ == "GlbHandle" or tuple(node.color) != (255, 255, 255):
            raise SystemExit(f"1-channel texture plane drawn as {type(node).__name__}, not the base color")
        if not any(u["kind"] == "texture" and u["name"] == "gray" for u in viewer.scene_report()["unsupported"]):
            raise SystemExit("1-channel texture missing from unsupported report")
    renderer = mujoco.Renderer(model, 200, 200)
    renderer.update_scene(data, camera="top")
    shot = renderer.render()
    renderer.close()
    colors = {tuple(int(c) for c in shot[y, x]) for y in range(70, 130, 10) for x in range(70, 130, 10)}
    if colors != {(255, 255, 255)}:
        raise SystemExit(f"MuJoCo drew the 1-channel plane as {colors}, not the base color")
    return {"mj_viser": "base color, reported", "mujoco_renderer": "base color"}
