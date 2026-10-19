# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Image channels on a board, in recordings, and in live panels, without ssrobot.

The checks: a board's image channels are current, malformed (a reading that is not an
RGB image leaves no older image to show), or unavailable, as fed; a StateRecorder
refuses a non-image, records each image at its time, decoded to its colour, and hides
an image and shows it again; both browsers of a LiveView get both images; several
readings between two frames send one image while the board counts every reading; and
browsers connecting while every frame renders never make refresh raise. Writes
images.viser to the output directory.
"""

from __future__ import annotations

import io
import os
import threading
import time
from importlib.metadata import version
from typing import Any

import msgspec
import mujoco
import numpy as np
from PIL import Image
from websockets.sync.client import connect

from e2e.channels import CAMERA, _frames
from e2e.fixtures import expect_raises
from e2e.live_view import _Source
from e2e.oracles import replay_recording
from mj_viser import ChannelBoard, LiveView, StateRecorder
from mj_viser.recording import IMAGES

BROWSERS = 20
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0)}


def _image(colour: str, shape: tuple[int, int]) -> np.ndarray:
    pixels = np.empty((*shape, 3), np.uint8)
    pixels[:] = COLOURS[colour]
    return pixels


def _colour(jpeg: bytes) -> str:
    rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), float).reshape(-1, 3).mean(axis=0)
    return min(COLOURS, key=lambda c: float(np.abs(rgb - COLOURS[c]).sum()))


def check_images(model: mujoco.MjModel, out_dir: str) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # A board: two cameras, one more not provided, and a reading that is not an image.
    channels = ChannelBoard()
    for name in ("left_cam", "right_cam"):
        channels.declare(name, kind="image")
    channels.declare("aux_cam", kind="image", available=False)
    channels.reading("left_cam", 0, _image("red", (24, 32)))
    channels.reading("right_cam", 0, _image("blue", (16, 40)))
    channels.reading("right_cam", 300_000_000, np.zeros((16, 40), np.float32))
    channels.advance(300_000_000)
    states = {c.name: (c.status, c.text) for c in channels.channels()}
    expected = {
        "aux_cam": ("unavailable", "not provided"),
        "left_cam": ("current", "uint8 24×32×3 (t = 0.000 s)"),
        "right_cam": ("malformed", "malformed: float32 16×40, not uint8 H×W×3"),
    }
    if states != expected:
        raise SystemExit(f"image channel states {states}")
    if channels.image("right_cam")[0] is not None:
        raise SystemExit("a malformed reading left an older image to show")
    out["states"] = {name: status for name, (status, _) in states.items()}

    # The board keeps its own, read-only copy: a camera that reuses its buffer does not
    # change a reading already taken.
    buffer = _image("red", (24, 32))
    channels.reading("left_cam", 350_000_000, buffer)
    buffer[:] = COLOURS["green"]
    kept, _ = channels.image("left_cam")
    assert kept is not None
    if not np.array_equal(kept[0, 0], COLOURS["red"]):
        raise SystemExit("a reused camera buffer changed a reading already taken")
    out["kept_image_read_only"] = expect_raises(ValueError, lambda: kept.__setitem__((0, 0), 0))

    # A recording: each image at its time; a hidden image hides, and shows again.
    data = mujoco.MjData(model)
    with StateRecorder(model) as recorder:
        out["refused"] = {
            "float_image": expect_raises(ValueError, lambda: recorder.image("bad", np.zeros((4, 4, 3), np.float32))),
            "list": expect_raises(ValueError, lambda: recorder.image("bad", [[0, 0, 0]])),  # type: ignore[arg-type]
            "empty_width": expect_raises(ValueError, lambda: recorder.image("bad", np.zeros((4, 0, 3), np.uint8))),
        }
        recorder.advance(0)
        recorder.show(data)
        recorder.image("left_cam", _image("red", (24, 32)))
        recorder.image("right_cam", _image("blue", (16, 40)))
        recorder.advance(100_000_000)
        recorder.image("left_cam", _image("green", (24, 32)))
        recorder.advance(200_000_000)
        recorder.image("right_cam", _image("yellow", (16, 40)))
        recorder.advance(300_000_000)
        recorder.image("right_cam", None)
        recorder.advance(400_000_000)
        recorder.image("right_cam", _image("blue", (16, 40)))
        recording = recorder.finish()
    messages = replay_recording(recording)
    seen = []
    for t, m in messages:
        name = m.get("name", "")
        if name.startswith(IMAGES + "/"):
            image = (m.get("props") or m.get("updates") or {}).get("_data")
            if image is not None:
                seen.append((round(t, 6), name.rsplit("/", 1)[1], _colour(image)))
    want = [
        (0.0, "left_cam", "red"),
        (0.0, "right_cam", "blue"),
        (0.1, "left_cam", "green"),
        (0.2, "right_cam", "yellow"),
        (0.4, "right_cam", "blue"),
    ]
    if seen != want:
        raise SystemExit(f"recorded images {seen}")
    visibility = [
        (round(t, 6), m["visible"])
        for t, m in messages
        if m["type"] == "SetSceneNodeVisibilityMessage" and m.get("name") == f"{IMAGES}/right_cam" and t > 0
    ]
    if visibility != [(0.3, False), (0.4, True)]:
        raise SystemExit(f"right_cam visibility over the recording: {visibility}")
    if any(m.get("name", "") == f"{IMAGES}/bad" for _, m in messages):
        raise SystemExit("a refused image left a node in the recording")
    out["recorded"] = [f"{t} s {name} {colour}" for t, name, colour in seen]
    with open(os.path.join(out_dir, "images.viser"), "wb") as f:
        f.write(recording)

    # Live: both browsers get both images; readings between frames are coalesced.
    channels.reading("right_cam", 400_000_000, _image("blue", (16, 40)))
    view = LiveView(_Source(model), channels=channels, max_rate_hz=None, port=0)
    url = view.open().replace("http://", "ws://")
    clients = [connect(url, subprotocols=[f"viser-v{version('viser')}"], max_size=None) for _ in range(2)]
    try:
        for ws in clients:
            ws.send(msgspec.msgpack.encode(CAMERA))
        got = [
            sorted(m["props"]["label"] for m in _frames(ws, 1.0) if m["type"] == "GuiImageMessage") for ws in clients
        ]
        if got != [["left_cam", "right_cam"]] * 2:
            raise SystemExit(f"images per client: {got}")
        panels = [view._panels.client(i) for i in view.viewer.server.get_clients()]  # type: ignore[union-attr]
        before = [p.images_sent["left_cam"] for p in panels]
        for i in range(5):
            colour = "green" if i % 2 else "red"
            channels.reading("left_cam", 400_000_000 + i, _image(colour, (24, 32)))
        view.refresh(force=True)
        time.sleep(0.2)
        sent = [p.images_sent["left_cam"] - b for p, b in zip(panels, before, strict=True)]
        if sent != [1, 1] or channels.coverage()["left_cam"]["readings"] != 7:
            raise SystemExit(
                f"5 readings between frames sent {sent} images; model counted {channels.coverage()['left_cam']}"
            )
        out["live"] = {"images_per_client": got[0], "readings_between_frames": 5, "images_sent_per_client": 1}
    finally:
        for ws in clients:
            ws.close()
        view.close()

    # Browsers connect and leave while the simulation refreshes every frame: panels are
    # built on Viser's thread as frames render on this one, and refresh never raises.
    view = LiveView(_Source(model), channels=channels, max_rate_hz=None, port=0)
    url = view.open().replace("http://", "ws://")
    errors: list[BaseException] = []
    running = threading.Event()
    running.set()

    def loop() -> None:
        while running.is_set():
            try:
                view.refresh(force=True)
            except BaseException as e:  # noqa: BLE001 - recorded and reported below
                errors.append(e)
                return

    refresher = threading.Thread(target=loop)
    refresher.start()
    try:
        for _ in range(BROWSERS):
            ws = connect(url, subprotocols=[f"viser-v{version('viser')}"], max_size=None)
            ws.send(msgspec.msgpack.encode(CAMERA))
            _frames(ws, 0.2)
            ws.close()
    finally:
        running.clear()
        refresher.join()
        view.close()
    if errors:
        raise SystemExit(f"refresh raised while browsers connected: {errors[0]!r}")
    out["browsers_connect_during_refresh"] = BROWSERS
    return out
