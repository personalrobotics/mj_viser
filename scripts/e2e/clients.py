# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Browser-facing behavior: per-client selection and panel-free group visibility."""

from __future__ import annotations

import time
from importlib.metadata import version
from typing import Any

import msgspec
import mujoco
import zstandard

import mj_viser
from e2e.fixtures import MOCAP, STATES, kinematics
from e2e.oracles import check_against_mujoco_scene


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
