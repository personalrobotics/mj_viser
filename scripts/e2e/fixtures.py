# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The end-to-end test model, its two caller states, and small helpers."""

from __future__ import annotations

import hashlib
import socket

import mujoco
import numpy as np

MJCF = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="8" height="8" rgb1="1 1 1" rgb2="0.2 0.2 0.2"/>
    <material name="checkered" texture="checker"/>
    <material name="tinted" rgba="0.1 0.2 0.3 0.4"/>
    <material name="layered">
      <layer texture="checker" role="rgb"/>
      <layer texture="checker" role="emissive"/>
    </material>
    <hfield name="bumps" nrow="2" ncol="2" size="0.5 0.5 0.1 0.05" elevation="0 1 1 0"/>
    <mesh name="tet" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2" texcoord="0 0  1 0  0 1  1 1"
          face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <geom name="tet" type="mesh" mesh="tet" material="checkered" pos="-0.5 0 0"/>
    <!-- Infinite (size 0) textured plane: rendered as a large square, with a diagnostic. -->
    <geom name="floor" type="plane" size="0 0 0.1" material="checkered"/>
    <geom name="terrain" type="hfield" hfield="bumps" pos="3 3 0"/>
    <!-- Material rules: a texture on a box is not drawn; a material's alpha is kept. -->
    <geom name="painted" type="box" size="0.05 0.05 0.05" pos="2 0 0.5" material="checkered"/>
    <geom name="glass" type="sphere" size="0.05" pos="2.2 0 0.5" material="tinted"/>
    <geom name="egg" type="ellipsoid" size="0.05 0.08 0.1" pos="2.4 0 0.5" euler="0.3 0.2 0.1"/>
    <geom name="can" type="cylinder" size="0.05 0.1" pos="2.6 0 0.5" euler="0.5 0 0"/>
    <!-- Explicit color one ulp-ish from the default: still explicit, so it beats the material. -->
    <geom name="almost" type="box" size="0.03 0.03 0.03" pos="2.8 0 0.5" material="tinted" rgba="0.5000001 0.5 0.5 1"/>
    <!-- Layered material: the rgb role is drawn, the emissive role is reported. -->
    <geom name="layeredplane" type="plane" size="0.2 0.2 0.1" pos="3.5 0 0.2" material="layered"/>
    <light name="lamp" pos="0 0 3"/>
    <camera name="overhead" pos="0 0 3"/>
    <!-- "ns/a_b" and "ns_a/b" sanitize to the same text; the geom in the first is unnamed. -->
    <body name="ns/a_b" pos="1 0 0"><geom type="sphere" size="0.05"/></body>
    <body name="ns_a/b" pos="1.2 0 0"><geom name="twin" type="sphere" size="0.05"/></body>
    <!-- Far below the floor: must stay visible (no height heuristic). -->
    <body name="deep" pos="0 1 -2"><geom name="below" type="sphere" size="0.1"/></body>
    <!-- Mocap: welded to the world, yet moved by mocap_pos/mocap_quat. -->
    <body name="target" mocap="true" pos="0 -1 0.5"><geom name="target_box" type="box" size="0.05 0.05 0.05"/></body>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="link" type="capsule" size="0.05" fromto="0 0 0 0.3 0 0" rgba="0.8 0.3 0.3 1"/>
      <!-- One site per shape; one unnamed, one transparent (filtered), one in group 4 (hidden by default). -->
      <site name="mark_capsule" type="capsule" size="0.01 0.03" pos="0.1 0 0.06" rgba="1 0.5 0 1"/>
      <site name="mark_cylinder" type="cylinder" size="0.01 0.02" pos="0.15 0 0.06" euler="0.4 0 0"/>
      <site name="mark_box" type="box" size="0.01 0.015 0.02" pos="0.2 0 0.06" material="tinted"/>
      <site name="mark_ellipsoid" type="ellipsoid" size="0.01 0.015 0.02" pos="0.25 0 0.06" group="4"/>
      <site type="sphere" size="0.01" pos="0.05 0 0.06"/>
      <site name="ghost" size="0.01" pos="0.3 0 0.06" rgba="1 0 0 0"/>
      <!-- A texture on a site's material is not drawn (as in MuJoCo's classic renderer) and is reported. -->
      <site name="painted_site" type="box" size="0.01 0.01 0.01" pos="0.12 0 0.08" material="checkered"/>
      <body name="slider" pos="0.3 0 0">
        <joint name="slide" type="slide" axis="1 0 0"/>
        <!-- An explicit rgba overrides its material's. -->
        <geom name="tip" type="box" size="0.04 0.04 0.04" rgba="0.3 0.3 0.8 1" material="tinted"/>
        <site name="tip_site" pos="0.04 0 0"/>
        <!-- Fixed child of an articulated body: moves with it, but its local pose never changes. -->
        <body name="cap" pos="0.05 0 0"><geom name="cap_ball" type="sphere" size="0.02"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


STATES = {"A": [0.3, 0.1], "B": [-0.7, 0.25]}


MOCAP = {"A": [0.0, -1.0, 0.5], "B": [1.0, 2.0, 3.0]}  # position of the "target" mocap body


HASHED_FIELDS = ("time", "qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "xpos", "geom_xpos", "sensordata")


def state_hash(data: mujoco.MjData) -> str:
    h = hashlib.sha256()
    for field in HASHED_FIELDS:
        h.update(field.encode())
        h.update(np.ascontiguousarray(getattr(data, field)).tobytes())
    return h.hexdigest()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def kinematics(
    model: mujoco.MjModel, qpos: np.ndarray | list[float], mocap_pos: list[float] | None = None
) -> mujoco.MjData:
    # Independent reference: MuJoCo kinematics on a fresh MjData.
    ref = mujoco.MjData(model)
    ref.qpos[:] = qpos
    if mocap_pos is not None:
        ref.mocap_pos[0] = mocap_pos
    mujoco.mj_kinematics(model, ref)
    return ref


def expect_raises(exc: type[BaseException], fn) -> str:  # type: ignore[no-untyped-def]
    try:
        fn()
    except exc as e:
        return f"{type(e).__name__}: {e}"
    raise SystemExit(f"expected {exc.__name__} from {fn}")


def source_states(model: mujoco.MjModel) -> dict[str, mujoco.MjData]:
    """The caller's two states. State B skips kinematics: the viewer must compute them itself."""
    sources = {}
    for name, qpos in STATES.items():
        d = mujoco.MjData(model)
        d.qpos[:] = qpos
        d.mocap_pos[0] = MOCAP[name]
        d.time = {"A": 1.0, "B": 2.0}[name]
        if name == "A":
            mujoco.mj_forward(model, d)  # B deliberately skips kinematics: the viewer must compute them itself.
        sources[name] = d

    return sources
