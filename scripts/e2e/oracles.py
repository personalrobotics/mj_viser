# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Independent oracles: MuJoCo's own kinematics and scene, the .viser format, and glTF."""

from __future__ import annotations

import html.parser
import io
import json
import struct
from typing import Any

import msgspec
import mujoco
import numpy as np
import PIL.Image
import zstandard

import mj_viser

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


def same_rotation(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    return bool(np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol))


def check_world_poses(
    local: dict[str, Pose], scene: dict, model: mujoco.MjModel, data: mujoco.MjData, where: str, atol: float = 1e-6
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
        if not (np.allclose(pos, exp_pos, atol=atol) and same_rotation(quat, exp_quat, atol)):
            raise SystemExit(f"{where}: {path} at {pos}, {quat}; MuJoCo has {exp_pos}, {exp_quat}")
        checked += 1
    return checked


def replay_recording(recording: bytes) -> list[tuple[float, dict]]:  # type: ignore[type-arg]
    """Decode a .viser file the way the Viser client does: u64 size, zstd body of
    [u64 msgpack length][msgpack {messages: [[time, message], ...]}][binary buffers]."""
    inner_size = int.from_bytes(recording[:8], "little")
    inner = zstandard.ZstdDecompressor().decompress(recording[8:], max_output_size=inner_size)
    n = int.from_bytes(inner[:8], "little")
    payload = msgspec.msgpack.decode(inner[8 : 8 + n])
    return [(float(t), m) for t, m in payload["messages"]]


def replay_arrays(recording: bytes) -> list[tuple[float, dict]]:  # type: ignore[type-arg]
    """As :func:`replay_recording`, with each binary placeholder (``{"__binary_index":
    i, "dtype": ...}``) replaced by its flat numpy array: buffers follow the msgpack
    payload, each 8-byte aligned, with their lengths in ``binaryBufferLengths``."""
    inner_size = int.from_bytes(recording[:8], "little")
    inner = zstandard.ZstdDecompressor().decompress(recording[8:], max_output_size=inner_size)
    n = int.from_bytes(inner[:8], "little")
    payload = msgspec.msgpack.decode(inner[8 : 8 + n])
    buffers, offset = [], 8 + n
    for length in payload.get("binaryBufferLengths", ()):
        offset += (8 - offset % 8) % 8
        buffers.append(inner[offset : offset + length])
        offset += length

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            if "__binary_index" in value and "dtype" in value:
                return np.frombuffer(buffers[value["__binary_index"]], dtype=np.dtype(value["dtype"]))
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return [(float(t), resolve(m)) for t, m in payload["messages"]]


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
