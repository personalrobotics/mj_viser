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
import sys
import time
import tokenize
import urllib.parse
from importlib.metadata import version
from typing import Any

import msgspec
import mujoco
import numpy as np
import zstandard

import mj_viser

# A startup hang is one of the failures under test; never let it stall CI.
faulthandler.dump_traceback_later(60, exit=True)

MJCF = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="8" height="8" rgb1="1 1 1" rgb2="0.2 0.2 0.2"/>
    <material name="checkered" texture="checker"/>
    <mesh name="tet" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2" texcoord="0 0  1 0  0 1  1 1"
          face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <geom name="tet" type="mesh" mesh="tet" material="checkered" pos="-0.5 0 0"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <light name="lamp" pos="0 0 3"/>
    <camera name="overhead" pos="0 0 3"/>
    <!-- "ns/a_b" and "ns_a/b" sanitize to the same text; the geom in the first is unnamed. -->
    <body name="ns/a_b" pos="1 0 0"><geom type="sphere" size="0.05"/></body>
    <body name="ns_a/b" pos="1.2 0 0"><geom name="twin" type="sphere" size="0.05"/></body>
    <!-- Far below the floor: must stay visible (no height heuristic). -->
    <body name="deep" pos="0 1 -2"><geom name="below" type="sphere" size="0.1"/></body>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="link" type="capsule" size="0.05" fromto="0 0 0 0.3 0 0" rgba="0.8 0.3 0.3 1"/>
      <body name="slider" pos="0.3 0 0">
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom name="tip" type="box" size="0.04 0.04 0.04" rgba="0.3 0.3 0.8 1"/>
        <site name="tip_site" pos="0.04 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

STATES = {"A": [0.3, 0.1], "B": [-0.7, 0.25]}

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
    nodes.update({report["sites"][i]["path"]: h for i, h in sm._site_frames.items()})
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


def kinematics(model: mujoco.MjModel, qpos: np.ndarray | list[float]) -> mujoco.MjData:
    # Independent reference: MuJoCo kinematics on a fresh MjData.
    ref = mujoco.MjData(model)
    ref.qpos[:] = qpos
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
    """Pose messages that touched a geom or site after the initial scene snapshot."""
    children = {g["path"] for g in scene["geoms"]} | {s["path"] for s in scene["sites"]}
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


def check_scene_report(scene: dict, model: mujoco.MjModel, viewer: mj_viser.MujocoViewer) -> None:  # type: ignore[type-arg]
    """Paths are unique and id-based; every element mj_viser skips is reported."""
    paths = [x["path"] for k in ("bodies", "geoms", "sites") for x in scene[k]]
    if len(set(paths)) != len(paths):
        raise SystemExit("scene paths collide")
    unnamed = [g for g in scene["geoms"] if not g["name"]]
    if not unnamed or not all(g["path"].rsplit("/", 1)[-1] == f"g{g['id']}" for g in unnamed):
        raise SystemExit(f"unnamed geoms lack id paths: {unnamed}")
    twins = [b["path"] for b in scene["bodies"] if b["name"] in ("ns/a_b", "ns_a/b")]
    if len(set(twins)) != 2:
        raise SystemExit(f"names that sanitize alike share a path: {twins}")
    reported = {(u["kind"], u["name"]) for u in scene["unsupported"]}
    for expected in (("geom", "floor"), ("light", "lamp"), ("camera", "overhead"), ("site", "tip_site")):
        if expected not in reported:
            raise SystemExit(f"{expected} missing from unsupported report: {sorted(reported)}")
    live = live_local_poses(viewer)
    if missing := [
        p
        for p in paths
        if p not in live and not any(g["path"] == p and g["status"] != "rendered" for g in scene["geoms"])
    ]:
        raise SystemExit(f"reported paths with no scene node: {missing}")


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

    Writes robot_two_states.viser and robot_report.json (scene report plus body and child
    local transforms per state) and checks every body and geom, live and in the recording.
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

    label = "/".join(xml_path.replace(os.sep, "/").split("/")[-2:])  # machine-independent
    out: dict[str, Any] = {"model": label, "states": {}}
    with mj_viser.MujocoViewer(model, port=0) as viewer:
        scene = viewer.scene_report()
        handles = viewer._scene_mgr._geom_handles  # type: ignore[union-attr]
        out["textured_mesh_nodes"] = sum(type(h).__name__ == "GlbHandle" for h in handles.values())
        if out["textured_mesh_nodes"] == 0:
            raise SystemExit(f"{xml_path} rendered no textured meshes")
        child_paths = {g["path"] for g in scene["geoms"]} | {x["path"] for x in scene["sites"]}
        serializer = viewer.server.get_scene_serializer()
        children: dict[str, Any] = {}
        for name, qpos in states.items():
            d = mujoco.MjData(model)
            d.qpos[:] = qpos
            viewer.update(d)
            live = live_local_poses(viewer)
            checked = check_world_poses(live, scene, model, kinematics(model, qpos), f"robot live {name}")
            local = {p: {"position": v[0].round(9).tolist(), "wxyz": v[1].round(9).tolist()} for p, v in live.items()}
            state_children = {p: v for p, v in local.items() if p in child_paths}
            if children and state_children != children:
                raise SystemExit("robot geom or site local transforms changed between states")
            children = state_children
            out["states"][name] = {
                "qpos": qpos.round(9).tolist(),
                "world_poses_checked": checked,
                "body_local_transforms": {b["path"]: local[b["path"]] for b in scene["bodies"]},
            }
            serializer.insert_sleep(0.5)
        recording = serializer.serialize()
    messages = replay_recording(recording)
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

    with open(os.path.join(out_dir, "robot_two_states.viser"), "wb") as f:
        f.write(recording)
    out["child_local_transforms"] = children
    out["scene"] = scene
    out["recording"] = {"file": "robot_two_states.viser", "sha256": hashlib.sha256(recording).hexdigest()}
    with open(os.path.join(out_dir, "robot_report.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    return {
        "model": label,
        "bodies": len(scene["bodies"]),
        "world_poses_checked": {k: v["world_poses_checked"] for k, v in out["states"].items()},
        "unsupported": sorted(f"{u['kind']}:{u['name']}" for u in scene["unsupported"]),
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
        checked = check_world_poses(live, scene, model, kinematics(model, STATES[name]), f"live state {name}")
        if viewer.data is d:
            raise SystemExit("viewer retained the caller's MjData")
        states_report[name] = {
            "qpos": STATES[name],
            "source_sha256_before": before,
            "source_sha256_after": after,
            "world_poses_checked": checked,
        }
        serializer.insert_sleep(0.5)
    recording = serializer.serialize()
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
        check_world_poses(recorded, scene, model, kinematics(model, STATES[name]), f"recording state {name}")
    if moved := static_child_messages(messages, scene):
        raise SystemExit(f"recording moved static children: {moved[:5]}")
    report["recording"]["replays_to_states"] = ["A", "B"]

    report["two_client_selection"] = two_client_selection(model)

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
    if robot := os.environ.get("MJ_VISER_ROBOT_XML"):
        report["robot"] = robot_scenario(robot, out_dir)

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
