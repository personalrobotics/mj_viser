# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Channel boards and their per-browser panels, without ssrobot.

A board declares four channels: joint positions with element names, a pose, joint
velocities that are not provided, and an unsupported depth image. It gets readings,
status lines (one carrying markup), and an event. The checks: each channel is
displayed, unavailable, or unsupported as declared; readings are shown by element name
with their time; status lines keep first-set order and can be removed; a channel goes
stale when the board's time passes without a reading; markup is escaped in panels; and
with two browsers connected to a LiveView, the channel one browser picks to plot is
plotted for it alone.
"""

from __future__ import annotations

import time
from importlib.metadata import version
from typing import Any

import msgspec
import mujoco
import numpy as np
import zstandard
from websockets.sync.client import connect

from e2e.fixtures import expect_raises
from e2e.live_view import _Source
from e2e.oracles import tags_in
from mj_viser import ChannelBoard, LiveView


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
    board = ChannelBoard(stale_after_ns=1_000_000_000)
    board.declare("arm_q", quantity="joint_position", labels=("shoulder", "elbow"))
    board.declare("arm_qd", quantity="joint_velocity", available=False)
    board.declare("hand_pose", quantity="pose", labels=("x", "y", "z", "qw", "qx", "qy", "qz"))
    board.declare("wrist_depth", kind="unsupported", quantity="depth_image", note="depth images are not shown")
    board.reading("arm_q", 0, (0.3, 0.1))
    board.reading("arm_q", 500_000_000, (-0.7, 0.25))
    board.reading("hand_pose", 500_000_000, (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0))
    board.set_status("task", "e1: <i>arm</i> · submitted → failed")  # markup is escaped where shown
    board.set_status("scene", "scene revision 1")
    board.event("opened")
    board.advance(600_000_000)
    coverage = {name: c["status"] for name, c in board.coverage().items()}
    expected = {"arm_q": "displayed", "arm_qd": "unavailable", "hand_pose": "displayed", "wrist_depth": "unsupported"}
    if coverage != expected:
        raise SystemExit(f"channel coverage {coverage}, expected {expected}")
    out["coverage"] = coverage
    lines = board.lines()
    for expect in (
        "t = 0.600 s",
        "e1: <i>arm</i> · submitted → failed",
        "scene revision 1",
        "arm_q = shoulder -0.700, elbow 0.250 (t = 0.500 s)",
        "hand_pose = x 0.100, y 0.200, z 0.300, qw 1.000, qx 0.000, qy 0.000, qz 0.000 (t = 0.500 s)",
        "opened",
    ):
        if expect not in lines:
            raise SystemExit(f"{expect!r} not among {lines}")
    if lines.index("e1: <i>arm</i> · submitted → failed") > lines.index("scene revision 1"):
        raise SystemExit(f"status lines out of first-set order: {lines}")
    board.set_status("scene", None)
    if "scene revision 1" in board.lines():
        raise SystemExit("a removed status line is still shown")
    board.advance(1_600_000_000)
    status = {c.name: c.status for c in board.channels()}
    if status["arm_q"] != "stale" or status["arm_qd"] != "unavailable":
        raise SystemExit(f"after 1.1 s without readings: {status}")
    out["stale_after_1s"] = status["arm_q"]

    # A channel's kind is part of its readings: changing it starts the channel over.
    board.declare("probe", kind="values")
    board.reading("probe", 1_600_000_000, (1.0, 2.0))
    board.declare("probe", kind="image")
    after_image = next(c for c in board.channels() if c.name == "probe")
    board.reading("probe", 1_600_000_000, np.zeros((2, 2, 3), np.uint8))
    board.declare("probe", kind="values")
    after_values = next(c for c in board.channels() if c.name == "probe")
    if (after_image.status, after_values.status) != ("awaiting", "awaiting"):
        raise SystemExit(f"after a kind change: {after_image}, then {after_values}")
    out["kind_change_starts_over"] = True
    out["unknown_kind"] = expect_raises(ValueError, lambda: board.declare("probe", kind="vector"))  # type: ignore[arg-type]
    channels = board

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
