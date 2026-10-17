# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""End-to-end check of an installed mj_viser wheel. Run by scripts/smoke_wheel.sh.

Imports every mj_viser module, then exercises the viewer lifecycle: renders two
deterministic caller-owned states and records them to a .viser file, proves the
source states are unchanged, and checks loopback default, port 0, repeated
open/close, and startup/update failures. Writes a JSON report and the recording
into the directory given as argv[1]; argv[2] names the environment in the report.
"""

from __future__ import annotations

import faulthandler
import hashlib
import html.parser
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import pkgutil
import socket
import struct
import sys
import time
import tokenize
import urllib.parse
from importlib.metadata import version
from typing import Any

import msgspec
import mujoco
import numpy as np
import PIL.Image
import zstandard

import mj_viser

# A startup hang is one of the failures under test; never let it stall CI.
faulthandler.dump_traceback_later(300, exit=True)

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

# Fields an update could plausibly touch: state, controls, derived kinematics, sensors.
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


Pose = tuple[np.ndarray, np.ndarray]  # (position, wxyz) in the parent's frame or the world


def compose(local: dict[str, Pose], path: str) -> Pose:
    """World pose of ``path`` from the local poses of it and its ancestors (missing = identity)."""
    pos, quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    parts = path.split("/")
    for i in range(2, len(parts) + 1):
        node = local.get("/".join(parts[:i]))
        if node is None:
            continue
        offset = np.zeros(3)
        mujoco.mju_rotVecQuat(offset, node[0], quat)
        combined = np.zeros(4)
        mujoco.mju_mulQuat(combined, quat, node[1])
        pos, quat = pos + offset, combined
    return pos, quat


def live_local_poses(viewer: mj_viser.MujocoViewer) -> dict[str, Pose]:
    # The poses on the Viser scene nodes themselves: what the server sends browsers.
    sm = viewer._scene_mgr  # type: ignore[union-attr]
    report = sm.report
    nodes = {report["bodies"][b]["path"]: h for b, h in sm._body_frames.items()}
    nodes.update({report["geoms"][g]["path"]: h for g, h in sm._geom_handles.items()})
    nodes.update({report["sites"][i]["path"]: h for i, h in sm._site_handles.items()})
    return {path: (np.asarray(h.position, float), np.asarray(h.wxyz, float)) for path, h in nodes.items()}


def recorded_local_poses(messages: list[tuple[float, dict]], before: float) -> dict[str, Pose]:  # type: ignore[type-arg]
    """Local pose of every node after applying the recording's messages earlier than ``before``."""
    raw: dict[str, dict[str, list[float]]] = {}
    for t, m in messages:
        if t >= before:
            break
        if m["type"] == "SetPositionMessage":
            raw.setdefault(m["name"], {})["position"] = m["position"]
        elif m["type"] == "SetOrientationMessage":
            raw.setdefault(m["name"], {})["wxyz"] = m["wxyz"]
    return {
        path: (np.asarray(v.get("position", [0, 0, 0]), float), np.asarray(v.get("wxyz", [1, 0, 0, 0]), float))
        for path, v in raw.items()
    }


def same_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, atol=1e-6) or np.allclose(a, -b, atol=1e-6))


def check_world_poses(
    local: dict[str, Pose], scene: dict, model: mujoco.MjModel, data: mujoco.MjData, where: str
) -> int:  # type: ignore[type-arg]
    """Every rendered geom and every body must sit where MuJoCo's kinematics put it."""
    checked = 0
    targets = [(b["path"], data.xpos[b["id"]], data.xquat[b["id"]]) for b in scene["bodies"]]
    for g in scene["geoms"]:
        if g["status"] == "rendered":
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, data.geom_xmat[g["id"]])
            targets.append((g["path"], data.geom_xpos[g["id"]], quat))
    for path, exp_pos, exp_quat in targets:
        pos, quat = compose(local, path)
        if not (np.allclose(pos, exp_pos, atol=1e-6) and same_rotation(quat, exp_quat)):
            raise SystemExit(f"{where}: {path} at {pos}, {quat}; MuJoCo has {exp_pos}, {exp_quat}")
        checked += 1
    return checked


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


def replay_recording(recording: bytes) -> list[tuple[float, dict]]:  # type: ignore[type-arg]
    """Decode a .viser file the way the Viser client does: u64 size, zstd body of
    [u64 msgpack length][msgpack {messages: [[time, message], ...]}][binary buffers]."""
    inner_size = int.from_bytes(recording[:8], "little")
    inner = zstandard.ZstdDecompressor().decompress(recording[8:], max_output_size=inner_size)
    n = int.from_bytes(inner[:8], "little")
    payload = msgspec.msgpack.decode(inner[8 : 8 + n])
    return [(float(t), m) for t, m in payload["messages"]]


def static_child_messages(messages: list[tuple[float, dict]], scene: dict) -> list[str]:  # type: ignore[type-arg]
    """Pose messages that touched a geom, a site, or a static body after the initial snapshot."""
    children = {g["path"] for g in scene["geoms"]} | {s["path"] for s in scene["sites"]}
    children |= {b["path"] for b in scene["bodies"] if not b["dynamic"]}
    return [
        f"{t}: {m['type']} {m['name']}"
        for t, m in messages
        if t > 0 and m["type"] in ("SetPositionMessage", "SetOrientationMessage") and m.get("name") in children
    ]


class TagCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag)


def tags_in(markup: str) -> set[str]:
    collector = TagCollector()
    collector.feed(markup)
    return collector.tags


def frozen(local: dict[str, Pose]) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    return {p: (tuple(np.round(a, 9)), tuple(np.round(b, 9))) for p, (a, b) in local.items()}


def glb_primitive(glb: bytes) -> dict[str, Any]:
    """The first mesh primitive of a GLB, decoded per the glTF 2.0 spec."""
    jlen = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20 : 20 + jlen])
    blen = struct.unpack_from("<I", glb, 20 + jlen)[0]
    binary = glb[28 + jlen : 28 + jlen + blen]

    def accessor(i: int) -> np.ndarray:
        a = doc["accessors"][i]
        view = doc["bufferViews"][a["bufferView"]]
        width = {"VEC2": 2, "VEC3": 3, "SCALAR": 1}[a["type"]]
        dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
        arr = np.frombuffer(binary, dtype, a["count"] * width, view.get("byteOffset", 0) + a.get("byteOffset", 0))
        return arr.reshape(a["count"], width) if width > 1 else arr

    prim = doc["meshes"][0]["primitives"][0]
    material = doc["materials"][prim["material"]]
    pbr = material["pbrMetallicRoughness"]
    out = {
        "positions": accessor(prim["attributes"]["POSITION"]).astype(float),
        "triangles": accessor(prim["indices"]).reshape(-1, 3),
        "factor": np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1]), float),
        "alpha_mode": material.get("alphaMode", "OPAQUE"),
        "uv": None,
        "image": None,
    }
    if "TEXCOORD_0" in prim["attributes"] and "baseColorTexture" in pbr:
        out["uv"] = accessor(prim["attributes"]["TEXCOORD_0"]).astype(float)
        view = doc["bufferViews"][
            doc["images"][doc["textures"][pbr["baseColorTexture"]["index"]]["source"]]["bufferView"]
        ]
        start = view.get("byteOffset", 0)
        out["image"] = np.asarray(PIL.Image.open(io.BytesIO(binary[start : start + view["byteLength"]])).convert("RGB"))
    return out


def node_vertices(handle: Any) -> np.ndarray:
    if type(handle).__name__ == "GlbHandle":
        return glb_primitive(handle.glb_data)["positions"]
    return np.asarray(handle.vertices, float)


def mujoco_scene(
    model: mujoco.MjModel, data: mujoco.MjData, visible: dict[str, set[int]]
) -> dict[tuple[str, int], Any]:
    """What MuJoCo's own visualizer draws for each geom and site with the same visible groups."""
    opt = mujoco.MjvOption()
    for kind, field in (("geoms", opt.geomgroup), ("sites", opt.sitegroup)):
        field[:] = [int(g in visible[kind]) for g in range(len(field))]
    scn = mujoco.MjvScene(model, maxgeom=10000)
    mujoco.mjv_updateScene(model, data, opt, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    kinds = {int(mujoco.mjtObj.mjOBJ_GEOM): "geoms", int(mujoco.mjtObj.mjOBJ_SITE): "sites"}
    return {(kinds[int(g.objtype)], int(g.objid)): g for g in scn.geoms[: scn.ngeom] if int(g.objtype) in kinds}


def check_against_mujoco_scene(
    viewer: mj_viser.MujocoViewer, model: mujoco.MjModel, data: mujoco.MjData, where: str
) -> int:
    """Every geom and site MuJoCo draws is drawn alike: visibility, type, size, world pose, and rgba.

    The oracle is mjv_updateScene with the viewer's visible groups, which resolves
    materials, sizes, and visibility exactly as MuJoCo's viewer does, independently
    of mj_viser's conversion.
    """
    expected = mujoco_scene(model, data, viewer.visible_groups())
    scene = viewer.scene_report()
    sm = viewer._scene_mgr  # type: ignore[union-attr]
    live = live_local_poses(viewer)
    infinite = mj_viser.geom_builders.INFINITE_PLANE_HALF_SIZE
    entries = [("geoms", e, sm._geom_handles.get(e["id"])) for e in scene["geoms"]]
    entries += [("sites", e, sm._site_handles.get(e["id"])) for e in scene["sites"]]
    for kind, entry, handle in entries:
        g = (kind, entry["id"])
        drawn = handle is not None and handle.visible
        if g not in expected:
            if drawn:
                raise SystemExit(f"{where}: {entry['path']} is shown but MuJoCo does not draw it")
            continue
        if not drawn:
            if entry["status"] == "unsupported":
                continue  # reported, not drawn
            raise SystemExit(f"{where}: MuJoCo draws {entry['path']} but it is not shown")
        mj = expected[g]
        gtype, size, rgba = int(mj.type), np.asarray(mj.size, float), np.asarray(mj.rgba, float)
        kind = type(handle).__name__

        # Size, against the geom type's own definition of its size parameters.
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            ok = np.allclose(handle.dimensions, 2 * size, atol=1e-6)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            ok = np.isclose(handle.radius, size[0], atol=1e-6)
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ok = np.isclose(handle.radius, size[0], atol=1e-6) and np.isclose(handle.height, 2 * size[2], atol=1e-6)
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            v = node_vertices(handle)
            axial = np.clip(v[:, 2], -size[2], size[2])
            dist = np.linalg.norm(v - np.column_stack([np.zeros_like(axial), np.zeros_like(axial), axial]), axis=1)
            ok = np.allclose(dist, size[0], atol=1e-5) and np.isclose(v[:, 2].max(), size[2] + size[0], atol=1e-5)
        elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            ok = np.allclose(((node_vertices(handle) / size) ** 2).sum(axis=1), 1.0, atol=1e-5)
        elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            v = node_vertices(handle)
            half = np.where(size[:2] > 0, size[:2], infinite)
            ok = np.allclose(np.abs(v[:, :2]).max(axis=0), half, atol=1e-6) and np.allclose(v[:, 2], 0)
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[entry["id"]]
            ref = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            # mesh_vert and glTF positions are both float32: the point sets must be identical.
            got = {tuple(r) for r in node_vertices(handle).astype(np.float32)}
            ok = got == {tuple(r) for r in np.asarray(ref, np.float32)}
        else:
            raise SystemExit(f"{where}: no size oracle for {entry['type']}")
        if not ok:
            raise SystemExit(f"{where}: {entry['path']} ({entry['type']}) size differs from MuJoCo's {size}")

        # Color and transparency.
        if kind == "GlbHandle":
            glb = glb_primitive(handle.glb_data)
            ok = np.allclose(glb["factor"], rgba, atol=1e-6) and glb["alpha_mode"] == (
                "BLEND" if rgba[3] < 1 else "OPAQUE"
            )
        else:
            opacity = handle.opacity
            ok = tuple(handle.color) == tuple(round(c * 255) for c in rgba[:3]) and (
                opacity is None if rgba[3] >= 1 else opacity is not None and abs(opacity - rgba[3]) < 1e-6
            )
        if not ok:
            raise SystemExit(f"{where}: {entry['path']} color differs from MuJoCo's rgba {rgba}")

        # World pose.
        pos, quat = compose(live, entry["path"])
        mj_quat = np.zeros(4)
        mujoco.mju_mat2Quat(mj_quat, np.asarray(mj.mat, float).reshape(9))
        if not (np.allclose(pos, mj.pos, atol=1e-6) and same_rotation(quat, mj_quat)):
            raise SystemExit(f"{where}: {entry['path']} pose differs from MuJoCo's scene")
    return len(expected)


def check_scene_report(scene: dict, model: mujoco.MjModel, viewer: mj_viser.MujocoViewer) -> None:  # type: ignore[type-arg]
    """Paths are unique and id-based; every element mj_viser skips is reported."""
    paths = [x["path"] for k in ("bodies", "geoms", "sites") for x in scene[k]]
    if len(set(paths)) != len(paths):
        raise SystemExit("scene paths collide")
    for kind, prefix in (("geoms", "g"), ("sites", "s")):
        unnamed = [e for e in scene[kind] if not e["name"]]
        if not unnamed or not all(e["path"].rsplit("/", 1)[-1] == f"{prefix}{e['id']}" for e in unnamed):
            raise SystemExit(f"unnamed {kind} lack id paths: {unnamed}")
    site_status = {e["name"]: e["status"] for e in scene["sites"]}
    if site_status.get("ghost") != "filtered" or site_status.get("mark_ellipsoid") != "rendered":
        raise SystemExit(f"site statuses wrong: {site_status}")
    twins = [b["path"] for b in scene["bodies"] if b["name"] in ("ns/a_b", "ns_a/b")]
    if len(set(twins)) != 2:
        raise SystemExit(f"names that sanitize alike share a path: {twins}")
    reported = {(u["kind"], u["name"]) for u in scene["unsupported"]}
    for expected in (
        ("geom", "terrain"),
        ("texture", "painted"),
        ("material-texture", "layered"),
        ("site-texture", "painted_site"),
        ("light", "lamp"),
        ("camera", "overhead"),
    ):
        if expected not in reported:
            raise SystemExit(f"{expected} missing from unsupported report: {sorted(reported)}")
    check_coverage(scene, model)
    if not any(d["name"] == "floor" and "infinite plane" in d["note"] for d in scene["diagnostics"]):
        raise SystemExit(f"infinite floor missing from diagnostics: {scene['diagnostics']}")
    live = live_local_poses(viewer)
    if missing := [
        p
        for p in paths
        if p not in live
        and not any(e["path"] == p and e["status"] != "rendered" for e in scene["geoms"] + scene["sites"])
    ]:
        raise SystemExit(f"reported paths with no scene node: {missing}")


def check_coverage(scene: dict, model: mujoco.MjModel) -> None:  # type: ignore[type-arg]
    """The coverage summary accounts for every visual element the model has."""
    totals = {
        "geom": model.ngeom,
        "site": model.nsite,
        "light": model.nlight,
        "camera": model.ncam,
        "tendon": model.ntendon,
        "skin": model.nskin,
        "flex": model.nflex,
    }
    for kind, total in totals.items():
        c = scene["coverage"][kind]
        if c["total"] != total or c["rendered"] + c["filtered"] + c["unsupported"] != total:
            raise SystemExit(f"coverage for {kind} does not account for all {total}: {c}")
        reported = sum(u["kind"] == kind for u in scene["unsupported"])
        if reported != c["unsupported"]:
            raise SystemExit(f"{kind}: {c['unsupported']} unsupported but {reported} listed")


def geom_visibility(viewer: mj_viser.MujocoViewer) -> dict[str, bool]:
    scene = viewer.scene_report()
    sm = viewer._scene_mgr  # type: ignore[union-attr]
    shown = {scene["geoms"][g]["path"]: bool(h.visible) for g, h in sm._geom_handles.items()}
    shown.update({scene["sites"][s]["path"]: bool(h.visible) for s, h in sm._site_handles.items()})
    return shown


def check_recorded_visibility(messages: list[tuple[float, dict]], live: dict[str, bool], where: str) -> None:  # type: ignore[type-arg]
    """Filtering is the same on replay: each geom's final recorded visibility equals the live one."""
    recorded: dict[str, bool] = {}
    for _, m in messages:
        if m["type"] == "SetSceneNodeVisibilityMessage":
            recorded[m["name"]] = bool(m["visible"])
    if differ := {p: (v, recorded.get(p)) for p, v in live.items() if recorded.get(p, True) != v}:
        raise SystemExit(f"{where}: visibility differs from live (live, recorded): {differ}")


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


def glb_color_at(glb: dict[str, Any], point: np.ndarray) -> np.ndarray:
    """Color glTF assigns at a point on the surface: texture (top-left origin, REPEAT) x baseColorFactor."""
    pos, uv, image = glb["positions"], glb["uv"], glb["image"]
    for tri in glb["triangles"]:
        a, b, c = pos[tri]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-12 or abs(np.dot(point - a, normal / np.linalg.norm(normal))) > 1e-4:
            continue
        w = np.linalg.lstsq(np.array([b - a, c - a]).T, point - a, rcond=None)[0]
        bary = np.array([1 - w.sum(), w[0], w[1]])
        if (bary >= -1e-6).all():
            u, v = (bary @ uv[tri]) % 1.0
            h, wdt = image.shape[:2]
            return image[min(int(v * h), h - 1), min(int(u * wdt), wdt - 1)] * glb["factor"][:3]
    raise SystemExit(f"no GLB triangle contains {point}")


def panel_free_groups(model: mujoco.MjModel) -> dict[str, Any]:
    """Group visibility is viewer state: it applies without the panel and before open().

    One panel-free viewer starts at MuJoCo's defaults (the group-4 site hidden) and then
    shows site group 4; another is configured before open(). Each state must match
    MuJoCo's view with the same groups. Malformed group ids are rejected and change nothing.
    """
    defaults = {"geoms": {0, 1, 2}, "sites": {0, 1, 2}}
    with_four = {"geoms": {0, 1, 2}, "sites": {0, 1, 2, 4}}
    at_a = kinematics(model, STATES["A"], MOCAP["A"])
    group4 = model.site("mark_ellipsoid").id
    out: dict[str, Any] = {}

    viewer = mj_viser.MujocoViewer(model, port=0, show_visibility=False)
    rejected = []
    for bad in ([1.9], ["2"], [True], [-1], [6], "012"):
        try:
            viewer.set_visible_groups(geoms=[0, 1], sites=bad)  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            rejected.append(f"{bad!r}: {type(e).__name__}")
        else:
            raise SystemExit(f"set_visible_groups accepted sites={bad!r}")
    if viewer.visible_groups() != defaults:
        raise SystemExit(f"rejected input changed visibility: {viewer.visible_groups()}")
    out["rejected"] = rejected
    with viewer:
        viewer.update(at_a)
        if viewer._gui_mgr is not None:  # type: ignore[attr-defined]
            raise SystemExit("show_visibility=False built a panel")
        check_against_mujoco_scene(viewer, model, at_a, "panel-free defaults")
        if viewer._scene_mgr._site_handles[group4].visible:  # type: ignore[union-attr]
            raise SystemExit("group-4 site shown by default without the panel")
        viewer.set_visible_groups(sites={0, 1, 2, 4})
        check_against_mujoco_scene(viewer, model, at_a, "panel-free, site group 4 shown")
    out["defaults_then_set"] = "matches MuJoCo"

    viewer = mj_viser.MujocoViewer(model, port=0, show_visibility=False)
    viewer.set_visible_groups(sites={0, 1, 2, 4})  # before open()
    if viewer.visible_groups() != with_four:
        raise SystemExit(f"before-open setting not kept: {viewer.visible_groups()}")
    with viewer:
        viewer.update(at_a)
        if not viewer._scene_mgr._site_handles[group4].visible:  # type: ignore[union-attr]
            raise SystemExit("site group 4, set before open(), is hidden")
        check_against_mujoco_scene(viewer, model, at_a, "panel-free, configured before open")
    out["configured_before_open"] = "matches MuJoCo"
    return out


def two_client_selection(model: mujoco.MjModel) -> dict[str, str]:
    """Two browser connections click different geoms; each sees only its own selection.

    Speaks Viser's wire protocol directly: a camera message registers each client,
    a SceneNodeClickMessage clicks, and server frames are zstd-compressed msgpack.
    """
    from websockets.sync.client import connect

    def frames(ws: Any, seconds: float) -> list[dict]:  # type: ignore[type-arg]
        out, end = [], time.time() + seconds
        while time.time() < end:
            try:
                raw = ws.recv(timeout=max(0.01, end - time.time()))
            except TimeoutError:
                break
            if isinstance(raw, bytes):
                size, csize = int.from_bytes(raw[:8], "little"), int.from_bytes(raw[8:16], "little")
                body = zstandard.ZstdDecompressor().decompress(raw[16 : 16 + csize], max_output_size=size)
                out.extend(msgspec.msgpack.decode(body)["messages"])
        return out

    camera = {
        "type": "ViewerCameraMessage",
        "wxyz": [1, 0, 0, 0],
        "position": [3, 0, 1],
        "fov": 1.0,
        "near": 0.01,
        "far": 100.0,
        "image_height": 480,
        "image_width": 640,
        "look_at": [0, 0, 0],
        "up_direction": [0, 0, 1],
    }
    selected: list[tuple[int, str]] = []
    with mj_viser.MujocoViewer(model, port=0) as viewer:
        viewer.on_select(lambda g, body: selected.append((g, body)))
        scene = viewer.scene_report()
        targets = {
            "first": scene["geoms"][model.geom("link").id]["path"],
            "second": scene["geoms"][model.geom("tip").id]["path"],
        }
        ws_url = viewer.url.replace("http://", "ws://")
        clients = {k: connect(ws_url, subprotocols=[f"viser-v{version('viser')}"]) for k in targets}
        try:
            for ws in clients.values():
                ws.send(msgspec.msgpack.encode(camera))
            for ws in clients.values():
                frames(ws, 1.0)
            for key, ws in clients.items():
                ws.send(
                    msgspec.msgpack.encode(
                        {
                            "type": "SceneNodeClickMessage",
                            "name": targets[key],
                            "instance_index": None,
                            "ray_origin": [0, 0, 0],
                            "ray_direction": [0, 0, 1],
                            "screen_pos": [0, 0],
                            "modifier": None,
                            "owner": "",
                        }
                    )
                )
            labels = {
                k: {m["name"] for m in frames(ws, 1.5) if m.get("type") == "LabelMessage"} for k, ws in clients.items()
            }
        finally:
            for ws in clients.values():
                ws.close()
    for key, own in targets.items():
        other = targets["second" if key == "first" else "first"]
        if not any(name.startswith(own + "/") for name in labels[key]) or any(
            name.startswith(other + "/") for name in labels[key]
        ):
            raise SystemExit(f"client {key} saw labels {labels[key]}; expected only under {own}")
    if sorted(g for g, _ in selected) != sorted([model.geom("link").id, model.geom("tip").id]):
        raise SystemExit(f"on_select saw {selected}")
    return {k: sorted(v)[0] for k, v in labels.items()}


def robot_scenario(xml_path: str, out_dir: str) -> dict[str, Any]:
    """A real articulated robot at two known states: its home keyframe, then every hinge bent 0.3 rad.

    Writes robot_<model>.viser and robot_<model>.json (scene report plus body and child
    local transforms per state, and which sites show) and checks every body, geom, and
    site, live and in the recording. Sites are then shown in every group, as an
    application would with ``set_visible_groups``, and must all match MuJoCo.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    home = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, home, 0)
    bent = home.qpos.copy()
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            bent[model.jnt_qposadr[j]] += 0.3
    states = {"A": home.qpos.copy(), "B": bent}

    label = "/".join(xml_path.replace(os.sep, "/").split("/")[-3:])  # machine-independent
    out: dict[str, Any] = {"model": label, "states": {}}
    with mj_viser.MujocoViewer(model, port=0) as viewer:
        scene = viewer.scene_report()
        handles = viewer._scene_mgr._geom_handles  # type: ignore[union-attr]
        out["textured_mesh_nodes"] = sum(type(h).__name__ == "GlbHandle" for h in handles.values())
        child_paths = {g["path"] for g in scene["geoms"]} | {x["path"] for x in scene["sites"]}
        serializer = viewer.server.get_scene_serializer()
        children: dict[str, Any] = {}
        for name, qpos in states.items():
            d = mujoco.MjData(model)
            d.qpos[:] = qpos
            viewer.update(d)
            live = live_local_poses(viewer)
            checked = check_world_poses(live, scene, model, kinematics(model, qpos), f"robot live {name}")
            drawn = check_against_mujoco_scene(viewer, model, kinematics(model, qpos), f"robot live {name}")
            local = {p: {"position": v[0].round(9).tolist(), "wxyz": v[1].round(9).tolist()} for p, v in live.items()}
            state_children = {p: v for p, v in local.items() if p in child_paths}
            if children and state_children != children:
                raise SystemExit("robot geom or site local transforms changed between states")
            children = state_children
            out["states"][name] = {
                "qpos": qpos.round(9).tolist(),
                "world_poses_checked": checked,
                "mujoco_scene_geoms_matched": drawn,
                "body_local_transforms": {b["path"]: local[b["path"]] for b in scene["bodies"]},
            }
            serializer.insert_sleep(0.5)
        recording = serializer.serialize()
        live_visible = geom_visibility(viewer)

        site_handles = viewer._scene_mgr._site_handles  # type: ignore[union-attr]

        def shown_sites() -> list[str]:
            return sorted(
                e["name"] or e["path"]
                for e in scene["sites"]
                if e["id"] in site_handles and site_handles[e["id"]].visible
            )

        by_default = shown_sites()
        viewer.set_visible_groups(sites=range(6))
        check_against_mujoco_scene(viewer, model, kinematics(model, states["B"]), f"{label} with all site groups")
        with_all = shown_sites()
        if hidden := [e["name"] for e in scene["sites"] if e["status"] == "rendered" and e["id"] not in site_handles]:
            raise SystemExit(f"{label}: rendered sites without nodes: {hidden}")
        out["sites"] = {"shown_by_default": by_default, "shown_with_all_site_groups": with_all}
    messages = replay_recording(recording)
    check_coverage(scene, model)
    check_recorded_visibility(messages, live_visible, "robot recording")
    for name, window_end in (("A", 0.5), ("B", float("inf"))):
        check_world_poses(
            recorded_local_poses(messages, window_end),
            scene,
            model,
            kinematics(model, states[name]),
            f"robot recording {name}",
        )
    if moved := static_child_messages(messages, scene):
        raise SystemExit(f"robot recording moved static children: {moved[:5]}")

    slug = label.removesuffix(".xml").replace("/", "_")
    with open(os.path.join(out_dir, f"robot_{slug}.viser"), "wb") as f:
        f.write(recording)
    out["child_local_transforms"] = children
    out["scene"] = scene
    out["recording"] = {"file": f"robot_{slug}.viser", "sha256": hashlib.sha256(recording).hexdigest()}
    with open(os.path.join(out_dir, f"robot_{slug}.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    return {
        "model": label,
        "bodies": len(scene["bodies"]),
        "world_poses_checked": {k: v["world_poses_checked"] for k, v in out["states"].items()},
        "unsupported": sorted(f"{u['kind']}:{u['name']}" for u in scene["unsupported"]),
        "coverage": scene["coverage"],
        "textured_mesh_nodes": out["textured_mesh_nodes"],
        "sites": {k: len(v) for k, v in out["sites"].items()},
        "recording_sha256": out["recording"]["sha256"],
    }


def expect_raises(exc: type[BaseException], fn) -> str:  # type: ignore[no-untyped-def]
    try:
        fn()
    except exc as e:
        return f"{type(e).__name__}: {e}"
    raise SystemExit(f"expected {exc.__name__} from {fn}")


def main(out_dir: str, env_label: str) -> None:
    modules = sorted(m.name for m in pkgutil.walk_packages(mj_viser.__path__, "mj_viser."))
    for name in modules:
        importlib.import_module(name)
    forbidden = [m for m in sys.modules if m.split(".")[0] == "mj_manipulator"]
    if forbidden:
        raise SystemExit(f"mj_viser imported robot-control modules: {forbidden}")

    # The viewer code (not its docstrings) must contain no stepping, reset, or worker thread of its own.
    # Nor may it reach into Viser's private modules or transport (viser._x, ._websock_*).
    names = set()
    private_viser = []
    for m in modules:
        src = io.StringIO(inspect.getsource(sys.modules[m]))
        toks = [t.string for t in tokenize.generate_tokens(src.readline) if t.type in (tokenize.NAME, tokenize.OP)]
        names |= set(toks)
        private_viser += [
            f"{m}: viser.{c}"
            for a, b, c in zip(toks, toks[1:], toks[2:])
            if a == "viser" and b == "." and c.startswith("_")
        ]
        private_viser += [f"{m}: {t}" for t in toks if t.startswith("_websock")]
    banned = sorted(names & {"mj_step", "mj_resetData", "Thread"})
    if banned:
        raise SystemExit(f"viewer lifecycle contains {banned}")
    if private_viser:
        raise SystemExit(f"private Viser access: {private_viser}")
    if any(hasattr(mj_viser.MujocoViewer, a) for a in ("launch", "launch_passive", "sync", "is_running")):
        raise SystemExit("legacy launch/sync API still present")

    model = mujoco.MjModel.from_xml_string(MJCF)
    report: dict[str, object] = {}

    # --- Two caller-owned states, rendered in order and recorded --------------------------
    sources = {}
    for name, qpos in STATES.items():
        d = mujoco.MjData(model)
        d.qpos[:] = qpos
        d.mocap_pos[0] = MOCAP[name]
        d.time = {"A": 1.0, "B": 2.0}[name]
        if name == "A":
            mujoco.mj_forward(model, d)  # B deliberately skips kinematics: the viewer must compute them itself.
        sources[name] = d

    viewer = mj_viser.MujocoViewer(model, port=0, label="mj_viser e2e")
    url = viewer.open()
    host, port = url.removeprefix("http://").rsplit(":", 1)
    if host != "127.0.0.1" or int(port) == 0:
        raise SystemExit(f"expected loopback URL with a real port, got {url}")
    report["default_url_host"] = host
    report["ephemeral_port_resolved"] = int(port) != 0

    # The textured mesh must take the trimesh + Pillow texture path (a GLB node), not the solid-color fallback.
    tet = viewer._scene_mgr._geom_handles[model.geom("tet").id]  # type: ignore[union-attr]
    if type(tet).__name__ != "GlbHandle":
        raise SystemExit(f"textured mesh rendered as {type(tet).__name__}, not a textured GLB")
    report["textured_mesh_node"] = type(tet).__name__

    scene = viewer.scene_report()
    check_scene_report(scene, model, viewer)
    report["scene"] = {
        "paths": len(scene["bodies"]) + len(scene["geoms"]) + len(scene["sites"]),
        "unsupported": sorted(f"{u['kind']}:{u['name']}" for u in scene["unsupported"]),
    }

    serializer = viewer.server.get_scene_serializer()
    states_report = {}
    child_paths = {g["path"] for g in scene["geoms"]} | {x["path"] for x in scene["sites"]}
    children_before = {p: v for p, v in live_local_poses(viewer).items() if p in child_paths}
    for name, d in sources.items():
        before = state_hash(d)
        viewer.update(d)
        after = state_hash(d)
        if before != after:
            raise SystemExit(f"update() modified source state {name}")
        live = live_local_poses(viewer)
        reference = kinematics(model, STATES[name], MOCAP[name])
        checked = check_world_poses(live, scene, model, reference, f"live state {name}")
        drawn = check_against_mujoco_scene(viewer, model, reference, f"live state {name}")
        if viewer.data is d:
            raise SystemExit("viewer retained the caller's MjData")
        states_report[name] = {
            "qpos": STATES[name],
            "source_sha256_before": before,
            "source_sha256_after": after,
            "world_poses_checked": checked,
            "mujoco_scene_geoms_matched": drawn,
        }
        serializer.insert_sleep(0.5)
    recording = serializer.serialize()
    live_visible = geom_visibility(viewer)

    # --- Group visibility is explicit viewer state ------------------------------------------
    # Showing every geom and site group must match MuJoCo with every group enabled, and
    # restoring MuJoCo's defaults (groups 0-2) must match MuJoCo's default view again.
    at_b = kinematics(model, STATES["B"], MOCAP["B"])
    viewer.set_visible_groups(geoms=range(6), sites=range(6))
    all_groups = check_against_mujoco_scene(viewer, model, at_b, "all groups shown")
    viewer.set_visible_groups(geoms={0, 1, 2}, sites={0, 1, 2})
    default_groups = check_against_mujoco_scene(viewer, model, at_b, "default groups restored")
    if all_groups <= default_groups:
        raise SystemExit("showing all groups revealed nothing; the group-4 site should appear")
    report["group_visibility"] = {"default_groups_drawn": default_groups, "all_groups_drawn": all_groups}

    # Geoms and sites are static in their body frames: updates must not have moved them.
    for path, (pos, quat) in children_before.items():
        if not (np.allclose(live[path][0], pos) and np.allclose(live[path][1], quat)):
            raise SystemExit(f"static child {path} changed between states")
    if below_visible := viewer._scene_mgr._geom_handles[model.geom("below").id].visible:  # type: ignore[union-attr]
        report["geom_below_floor_visible"] = below_visible
    else:
        raise SystemExit("geom below z=-0.5 was hidden")

    # --- Untrusted display strings are shown as text, never rendered as markup ------------
    payload = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
    viewer.set_hud("probe", payload)
    panel = mj_viser.SensorPanel(
        title=payload, channels=[mj_viser.SensorChannel(0, payload, color='red;"><img src=x>')], use_folder=False
    )
    viewer.add_panel(panel)
    markup = {"hud": viewer._huds["probe"].content, "sensor_legend": panel._legend.content}  # type: ignore[union-attr]
    for where, content in markup.items():
        injected = tags_in(content) - {"div", "span"}
        if injected:
            raise SystemExit(f"{where} rendered injected markup {sorted(injected)}: {content}")
    report["markup_escaped"] = sorted(markup)

    viewer.close()
    if viewer.is_open:
        raise SystemExit("viewer still open after close()")
    # The report is a function of the model alone: identical open, closed, or elsewhere.
    if viewer.scene_report() != scene or mj_viser.MujocoViewer(model).scene_report() != scene:
        raise SystemExit("scene report differs between viewers of the same model")
    with open(os.path.join(out_dir, "scene_report.json"), "w") as f:
        json.dump(scene, f, indent=2, sort_keys=True)
        f.write("\n")
    rec_path = os.path.join(out_dir, "two_states.viser")
    with open(rec_path, "wb") as f:
        f.write(recording)
    report["states"] = states_report
    report["recording"] = {"file": "two_states.viser", "sha256": hashlib.sha256(recording).hexdigest()}

    # --- The recording itself replays to both states ---------------------------------------
    # Oracle on the artifact users open, independent of the live server: replaying the
    # file's messages up to each state's timestamp yields MuJoCo's kinematics for that state.
    messages = replay_recording(recording)
    for name, window_end in (("A", 0.5), ("B", float("inf"))):
        recorded = recorded_local_poses(messages, window_end)
        check_world_poses(
            recorded, scene, model, kinematics(model, STATES[name], MOCAP[name]), f"recording state {name}"
        )
    if moved := static_child_messages(messages, scene):
        raise SystemExit(f"recording moved static children: {moved[:5]}")
    report["recording"]["replays_to_states"] = ["A", "B"]
    check_recorded_visibility(messages, live_visible, "recording")
    dynamic = {b["name"]: b["dynamic"] for b in scene["bodies"]}
    if not dynamic["target"] or dynamic["cap"]:
        raise SystemExit(f"mocap body must be dynamic and fixed child static: {dynamic}")
    report["dynamic_bodies"] = sorted(name for name, moves in dynamic.items() if moves)

    report["two_client_selection"] = two_client_selection(model)
    report["texture_oracle"] = texture_scenario()
    report["panel_free_groups"] = panel_free_groups(model)

    # --- Update failures return to the caller ---------------------------------------------
    # Adversarial models: every array the old size check compared is the same length.
    # "swapped" is the same layout with hinge and slide exchanged (0.5 m would render as 0.5 rad);
    # "camera" adds a camera, changing MjData's internal layout so mj_copyData would raise FatalError.
    adversarial = {
        "swapped": MJCF.replace('type="hinge"', 'type="TMP"')
        .replace('type="slide"', 'type="hinge"')
        .replace('type="TMP"', 'type="slide"'),
        "camera": MJCF.replace("<worldbody>", '<worldbody><camera name="cam" pos="0 0 2"/>'),
    }
    report["update_while_closed"] = expect_raises(RuntimeError, lambda: viewer.update(sources["A"]))
    with mj_viser.MujocoViewer(model, port=0) as v:
        v.update(sources["A"])
        good = frozen(live_local_poses(v))
        for name, xml in adversarial.items():
            m = mujoco.MjModel.from_xml_string(xml)
            if (m.nq, m.nv, m.nbody, m.ngeom, m.nsensordata) != (model.nq, model.nv, model.nbody, model.ngeom, 0):
                raise SystemExit(f"adversarial model {name} is not size-identical")
            d = mujoco.MjData(m)
            d.qpos[:] = [0.5, 0.5]
            report[f"update_{name}_model"] = expect_raises(ValueError, lambda: v.update(d))
            if frozen(live_local_poses(v)) != good:
                raise SystemExit(f"rejected {name} update changed the rendered scene")
        report["open_twice"] = expect_raises(RuntimeError, v.open)

    # --- IPv6 literal hosts produce a bracketed URL ---------------------------------------
    try:
        with socket.socket(socket.AF_INET6) as s6:
            s6.bind(("::1", 0))
    except OSError:
        report["ipv6_url"] = "skipped: ::1 not bindable on this host"
    else:
        with mj_viser.MujocoViewer(model, host="::1", port=0) as v:
            parsed = urllib.parse.urlsplit(v.url)
            if parsed.hostname != "::1" or not parsed.port:
                raise SystemExit(f"invalid IPv6 URL {v.url}")
            report["ipv6_url"] = v.url.replace(str(parsed.port), "<port>")

    # --- Repeated open/close on one fixed port releases it each time ----------------------
    fixed = free_port()
    v = mj_viser.MujocoViewer(model, port=fixed)
    for _ in range(3):
        if v.open() != f"http://127.0.0.1:{fixed}":
            raise SystemExit("reopen did not bind the requested port")
        v.update(sources["B"])
        v.close()
    v.close()  # idempotent
    with socket.socket() as s:
        s.bind(("127.0.0.1", fixed))  # port is free again
    report["open_close_cycles_on_fixed_port"] = 3

    # --- Startup failures return to the caller instead of hanging -------------------------
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        blocked = mj_viser.MujocoViewer(model, port=taken)
        report["startup_port_in_use"] = expect_raises(OSError, blocked.open).replace(str(taken), "<port>")
        if blocked.is_open:
            raise SystemExit("viewer reports open after failed startup")
    # 192.0.2.1 is TEST-NET-1 (RFC 5737): never assigned to a local interface.
    report["startup_unbindable_host"] = expect_raises(OSError, mj_viser.MujocoViewer(model, host="192.0.2.1").open)

    # --- A real textured articulated robot, when one is provided ---------------------------
    if robots := os.environ.get("MJ_VISER_ROBOT_XML"):
        report["robots"] = [robot_scenario(path, out_dir) for path in robots.split(os.pathsep)]
        if not any(r["textured_mesh_nodes"] for r in report["robots"]):
            raise SystemExit("no robot rendered a textured mesh")

    report.update(
        environment=env_label,
        installed={d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()},
        python=sys.version.split()[0],
        mj_viser=version("mj-viser"),
        mujoco=version("mujoco"),
        viser=version("viser"),
        mj_manipulator_installed=importlib.util.find_spec("mj_manipulator") is not None,
        modules_imported=modules,
        public_api=sorted(mj_viser.__all__),
    )
    with open(os.path.join(out_dir, "smoke_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
