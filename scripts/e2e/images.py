# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Image channels, live and in replay, without ssrobot or a renderer.

A stand-in description declares three RGB cameras; the runtime provides two. A stand-in
trace observes them independently with solid-colour images: left_cam red then green,
right_cam blue then yellow, then a right_cam reading that is not an RGB image, and then
time passes. The checks:

- replay: each well-formed image is in the recording at its recorded time, decoded to
  its colour; the report accounts for every image reading as shown or malformed; the
  label ends with left_cam stale and right_cam malformed; aux_cam is unavailable;
- live: both browsers get both images; several readings between two frames send one
  image (presentation throttling), while the model still counts every reading.

Writes images.viser and images_conversion.json to the output directory.
"""

from __future__ import annotations

import io
import json
import os
import time
from importlib.metadata import version
from types import SimpleNamespace as NS
from typing import Any

import msgspec
import mujoco
import numpy as np
from PIL import Image
from websockets.sync.client import connect

from e2e.channels import CAMERA, _frames
from e2e.fixtures import STATES
from e2e.live_view import _Source
from e2e.oracles import replay_recording
from e2e.replay import _Mapping, _Record, _snapshot
from mj_viser import ChannelModel, LiveView
from mj_viser.replay import LABEL, record_trace

DESCRIPTION = NS(
    groups=(),
    channels=tuple(NS(name=n, quantity="rgb_image", source=n) for n in ("aux_cam", "left_cam", "right_cam")),
)
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0)}


def _image(colour: str, shape: tuple[int, int]) -> NS:
    pixels = np.empty((*shape, 3), np.uint8)
    pixels[:] = COLOURS[colour]
    return NS(dtype="uint8", shape=pixels.shape, data=pixels.tobytes())


def _observed(t: int, **readings: NS) -> tuple[str, int, NS]:
    return ("observed", t, NS(readings=tuple(NS(channel=c, stamp=NS(time_ns=t), value=v) for c, v in readings.items())))


def _trace() -> list[_Record]:
    rows = [
        ("opened", 0, NS(runtime="stand-in", clock_mode="manual", channels=("left_cam", "right_cam"))),
        ("snapshot", 0, _snapshot(STATES["A"])),
        _observed(0, left_cam=_image("red", (24, 32)), right_cam=_image("blue", (16, 40))),
        _observed(100_000_000, left_cam=_image("green", (24, 32))),
        _observed(200_000_000, right_cam=_image("yellow", (16, 40))),
        _observed(300_000_000, right_cam=NS(dtype="float32", shape=(16, 40), data=bytes(16 * 40 * 4))),
        ("stepped", 1_500_000_000, None),
    ]
    return [_Record(i, k, "sim", t, p) for i, (k, t, p) in enumerate(rows)]


def _colour(jpeg: bytes) -> str:
    rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), float).reshape(-1, 3).mean(axis=0)
    return min(COLOURS, key=lambda c: float(np.abs(rgb - COLOURS[c]).sum()))


def check_images(model: mujoco.MjModel, out_dir: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    world = NS(model=model, world="w", description="d", mapping=_Mapping())
    replay = record_trace(_trace(), world, description=DESCRIPTION)
    report = replay.report
    expected = {
        "left_cam": {"readings": [0, 100_000_000], "shown": [0, 100_000_000], "malformed": []},
        "right_cam": {"readings": [0, 200_000_000, 300_000_000], "shown": [0, 200_000_000], "malformed": [300_000_000]},
    }
    if report["images"] != expected:
        raise SystemExit(f"image accounting {report['images']}, expected {expected}")
    coverage = {n: c["status"] for n, c in report["channels"].items()}
    if coverage != {"aux_cam": "unavailable", "left_cam": "displayed", "right_cam": "malformed"}:
        raise SystemExit(f"image channel coverage {coverage}")

    # The recording itself: each image's colour at its time, and the label's final state.
    messages = replay_recording(replay.recording)
    seen = []
    for t, m in messages:
        name = m.get("name", "")
        if name.startswith("/replay/images/"):
            data = (m.get("props") or m.get("updates") or {}).get("_data")
            if data is not None:
                seen.append((round(t, 6), name.rsplit("/", 1)[1], _colour(data)))
    if seen != [
        (0.0, "left_cam", "red"),
        (0.0, "right_cam", "blue"),
        (0.1, "left_cam", "green"),
        (0.2, "right_cam", "yellow"),
    ]:
        raise SystemExit(f"recorded images {seen}")
    texts = [(m.get("props") or m.get("updates") or {}).get("text") for _, m in messages if m.get("name") == LABEL]
    final = [s for s in texts if s is not None][-1].split("\n")
    for expect in (
        "left_cam = uint8 24×32×3 (t = 0.100 s, stale)",
        "right_cam = malformed: float32 16×40, not uint8 H×W×3",
    ):
        if expect not in final:
            raise SystemExit(f"{expect!r} not in the label's final state {final}")
    out["recorded"] = [f"{t} s {name} {colour}" for t, name, colour in seen]
    with open(os.path.join(out_dir, "images.viser"), "wb") as f:
        f.write(replay.recording)
    with open(os.path.join(out_dir, "images_conversion.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    # Live: both browsers get both images; readings between frames are coalesced.
    channels = ChannelModel(DESCRIPTION)
    for record in _trace()[:3]:
        channels(record)
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
            channels(
                _Record(
                    10 + i,
                    "observed",
                    "sim",
                    400_000_000 + i,
                    _observed(400_000_000 + i, left_cam=_image(colour, (24, 32)))[2],
                )
            )
        view.refresh(force=True)
        time.sleep(0.2)
        sent = [p.images_sent["left_cam"] - b for p, b in zip(panels, before, strict=True)]
        if sent != [1, 1] or channels.coverage()["left_cam"]["readings"] != 6:
            raise SystemExit(
                f"5 readings between frames sent {sent} images; model counted {channels.coverage()['left_cam']}"
            )
        out["live"] = {"images_per_client": got[0], "readings_between_frames": 5, "images_sent_per_client": 1}
    finally:
        for ws in clients:
            ws.close()
        view.close()
    return out
