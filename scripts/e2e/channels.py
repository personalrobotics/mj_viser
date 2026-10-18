# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Named channels, executions, and scene, live and per client, without ssrobot.

A stand-in description declares four channels: joint positions of a group, a pose, an
image, and joint velocities the runtime does not provide. A stand-in run opens, observes,
submits, and changes its scene. The checks: each channel is displayed, unavailable, or
unsupported as declared; readings are shown by joint name; a channel goes stale when
its reading ages; markup in trace text is escaped; and with two browsers connected to a
LiveView, the channel one browser picks to plot is plotted for it alone.
"""

from __future__ import annotations

import time
from importlib.metadata import version
from types import SimpleNamespace as NS
from typing import Any

import msgspec
import mujoco
import zstandard
from websockets.sync.client import connect

from e2e.live_view import _Source
from e2e.oracles import tags_in
from mj_viser import ChannelModel, LiveView

DESCRIPTION = NS(
    groups=(NS(name="arm", joints=("shoulder", "elbow")),),
    channels=(
        NS(name="arm_q", quantity="joint_position", source="arm"),
        NS(name="arm_qd", quantity="joint_velocity", source="arm"),
        NS(name="hand_pose", quantity="pose", source="hand"),
        NS(name="wrist_rgb", quantity="rgb_image", source="wrist_camera"),
    ),
)


def _reading(channel: str, t: int, value: Any) -> NS:
    return NS(channel=channel, stamp=NS(time_ns=t), value=value)


def _records() -> list[NS]:
    pose = (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0)
    rows = [
        ("opened", 0, NS(runtime="stand-in", clock_mode="manual", channels=("arm_q", "hand_pose", "wrist_rgb"))),
        ("observed", 0, NS(readings=(_reading("arm_q", 0, (0.3, 0.1)),))),
        ("submitted", 100_000_000, NS(execution="e1", command=NS(group="arm"))),
        ("status", 100_000_000, NS(execution="e1", state="active", diagnostic=None)),
        (
            "observed",
            500_000_000,
            NS(readings=(_reading("arm_q", 500_000_000, (-0.7, 0.25)), _reading("hand_pose", 500_000_000, pose))),
        ),
        ("status", 600_000_000, NS(execution="e1", state="failed", diagnostic=NS(code="<b>stalled</b>"))),
        ("scene", 600_000_000, NS(revision=1, attachments=(NS(object="cup", end_effector="hand", held=True),))),
    ]
    return [NS(sequence=i, kind=k, time_ns=t, payload=p) for i, (k, t, p) in enumerate(rows)]


def _frames(ws: Any, seconds: float) -> list[dict[str, Any]]:
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


CAMERA = {
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


def check_channels(model: mujoco.MjModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    channels = ChannelModel(DESCRIPTION, stale_after_ns=1_000_000_000)
    for record in _records():
        channels(record)  # the sink interface
    coverage = {name: c["status"] for name, c in channels.coverage().items()}
    expected = {"arm_q": "displayed", "arm_qd": "unavailable", "hand_pose": "displayed", "wrist_rgb": "unsupported"}
    if coverage != expected:
        raise SystemExit(f"channel coverage {coverage}, expected {expected}")
    out["coverage"] = coverage
    lines = channels.lines()
    for expect in (
        "arm_q = shoulder -0.700, elbow 0.250 (t = 0.500 s)",
        "hand_pose = x 0.100, y 0.200, z 0.300, qw 1.000, qx 0.000, qy 0.000, qz 0.000 (t = 0.500 s)",
        "e1: SimpleNamespace on arm · submitted → active → failed (<b>stalled</b>)",  # escaped in markup
        "scene revision 1: cup held by hand",
    ):
        if expect not in lines:
            raise SystemExit(f"{expect!r} not among {lines}")
    # Many executions: only the latest three are shown, so the panel keeps its size.
    for i in range(2, 6):
        channels(
            NS(
                sequence=90 + i,
                kind="submitted",
                time_ns=700_000_000,
                payload=NS(execution=f"e{i}", command=NS(group="<i>arm</i>" if i == 5 else "arm")),
            )
        )
    shown = [line for line in channels.lines() if line.startswith(("e", "…"))]
    if shown[0] != "…: 2 earlier" or [x.split(":")[0] for x in shown[1:]] != ["e3", "e4", "e5"]:
        raise SystemExit(f"executions shown: {shown}")
    out["executions_shown"] = shown[0]
    channels(NS(sequence=99, kind="stepped", time_ns=1_600_000_000, payload=None))
    status = {c.name: c.status for c in channels.channels()}
    if status["arm_q"] != "stale" or status["arm_qd"] != "unavailable":
        raise SystemExit(f"after 1.1 s without readings: {status}")
    out["stale_after_1s"] = status["arm_q"]

    # Live: two browsers, one picks a channel to plot.
    source = _Source(model)
    view = LiveView(source, channels=channels, max_rate_hz=None, port=0)
    url = view.open().replace("http://", "ws://")
    clients = [connect(url, subprotocols=[f"viser-v{version('viser')}"]) for _ in range(2)]
    try:
        for ws in clients:
            ws.send(msgspec.msgpack.encode(CAMERA))
        seen = [_frames(ws, 1.0) for ws in clients]
        dropdowns = [
            [m for m in frames if m["type"] == "GuiDropdownMessage" and m["props"]["label"] == "Channel"]
            for frames in seen
        ]
        if not all(len(d) == 1 for d in dropdowns):
            raise SystemExit(f"each client should get one channel dropdown: {dropdowns}")
        markup = [m["props"]["content"] for frames in seen for m in frames if m["type"] == "GuiHtmlMessage"]
        injected = set().union(*(tags_in(c) for c in markup)) - {
            "table",
            "tr",
            "td",
            "div",
            "span",
        }  # panels and the HUD
        if injected or not any("&lt;i&gt;arm&lt;/i&gt;" in c for c in markup):
            raise SystemExit(f"panel markup not escaped: {injected}")
        out["markup_escaped"] = True

        chosen = dropdowns[0][0]
        clients[0].send(
            msgspec.msgpack.encode({"type": "GuiUpdateMessage", "uuid": chosen["uuid"], "updates": {"value": "arm_q"}})
        )
        time.sleep(0.5)
        view.refresh(force=True)
        after = [_frames(ws, 1.0) for ws in clients]
        plots = [[m for m in frames if m["type"] == "GuiUplotMessage"] for frames in after]
        if len(plots[0]) != 1 or plots[1]:
            raise SystemExit(f"plots per client after the first chose arm_q: {[len(p) for p in plots]}")
        series = [s["label"] for s in plots[0][0]["props"]["series"]]
        if series != ["t (s)", "shoulder", "elbow"]:
            raise SystemExit(f"plot series {series}")
        strokes = [s.get("stroke") for s in plots[0][0]["props"]["series"][1:]]
        if not all(strokes) or len(set(strokes)) != len(strokes):
            raise SystemExit(f"plotted series must each draw a distinct line: {strokes}")
        out["per_client_plot"] = {"chooser": series, "other": "no plot"}
    finally:
        for ws in clients:
            ws.close()
        view.close()
    return out
