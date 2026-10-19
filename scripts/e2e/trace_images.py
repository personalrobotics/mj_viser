# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The ssrobot integration's image channels in replay, on stand-in records (needs ssrobot).

A stand-in description declares three RGB cameras; the runtime provides two. A stand-in
trace observes them independently with solid-colour images: left_cam red then green,
right_cam blue then yellow, then a reading that is not an RGB image, then blue again,
and then time passes. On the recording: each well-formed image is there at its recorded
time, decoded to its colour; the malformed reading hides right_cam's image while the
label says malformed, and the next valid reading shows it again; the report accounts
for every image reading as shown or malformed; both cameras end stale; aux_cam is
unavailable. Writes trace_images.viser and trace_images_conversion.json.
"""

from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace as NS
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from e2e.fixtures import STATES
from e2e.oracles import replay_recording
from e2e.trace_replay import _Mapping, _Record, _snapshot
from mj_viser.integrations.ssrobot.replay import record_trace
from mj_viser.recording import TEXT as LABEL

DESCRIPTION = NS(
    groups=(),
    channels=tuple(NS(name=n, quantity="rgb_image", source=n) for n in ("aux_cam", "left_cam", "right_cam")),
)
BROWSERS = 20
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
        _observed(400_000_000, right_cam=_image("blue", (16, 40))),
        ("stepped", 1_500_000_000, None),
    ]
    return [_Record(i, k, "sim", t, p) for i, (k, t, p) in enumerate(rows)]


def _colour(jpeg: bytes) -> str:
    rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), float).reshape(-1, 3).mean(axis=0)
    return min(COLOURS, key=lambda c: float(np.abs(rgb - COLOURS[c]).sum()))


def check_trace_images(model: mujoco.MjModel, out_dir: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    world = NS(model=model, world="w", description="d", mapping=_Mapping())
    replay = record_trace(_trace(), world, description=DESCRIPTION)
    report = replay.report
    expected = {
        "left_cam": {"readings": [0, 100_000_000], "shown": [0, 100_000_000], "malformed": []},
        "right_cam": {
            "readings": [0, 200_000_000, 300_000_000, 400_000_000],
            "shown": [0, 200_000_000, 400_000_000],
            "malformed": [300_000_000],
        },
    }
    if report["images"] != expected:
        raise SystemExit(f"image accounting {report['images']}, expected {expected}")
    coverage = {n: c["status"] for n, c in report["channels"].items()}
    if coverage != {"aux_cam": "unavailable", "left_cam": "displayed", "right_cam": "displayed"}:
        raise SystemExit(f"image channel coverage {coverage}")

    # The recording itself: each image's colour at its time, and the label's final state.
    messages = replay_recording(replay.recording)
    seen = []
    for t, m in messages:
        name = m.get("name", "")
        if name.startswith("/recording/images/"):
            data = (m.get("props") or m.get("updates") or {}).get("_data")
            if data is not None:
                seen.append((round(t, 6), name.rsplit("/", 1)[1], _colour(data)))
    if seen != [
        (0.0, "left_cam", "red"),
        (0.0, "right_cam", "blue"),
        (0.1, "left_cam", "green"),
        (0.2, "right_cam", "yellow"),
        (0.4, "right_cam", "blue"),
    ]:
        raise SystemExit(f"recorded images {seen}")
    # A malformed reading hides the image it cannot replace; the next valid one shows.
    visibility = [
        (round(t, 6), m["visible"])
        for t, m in messages
        if m["type"] == "SetSceneNodeVisibilityMessage" and m.get("name") == "/recording/images/right_cam" and t > 0
    ]
    if visibility != [(0.3, False), (0.4, True)]:
        raise SystemExit(f"right_cam visibility over the recording: {visibility}")
    labels = [
        (t, (m.get("props") or m.get("updates") or {}).get("text")) for t, m in messages if m.get("name") == LABEL
    ]
    labels = [(t, s.split("\n")) for t, s in labels if s is not None]

    def label_at(t: float) -> list[str]:
        return [s for when, s in labels if when <= t + 1e-9][-1]

    for t, expect in (
        (0.3, "right_cam = malformed: float32 16×40, not uint8 H×W×3"),
        (1.5, "left_cam = uint8 24×32×3 (t = 0.100 s, stale)"),
        (1.5, "right_cam = uint8 16×40×3 (t = 0.400 s, stale)"),
    ):
        if expect not in label_at(t):
            raise SystemExit(f"{expect!r} not in the label at {t} s: {label_at(t)}")
    out["recorded"] = [f"{t} s {name} {colour}" for t, name, colour in seen]
    with open(os.path.join(out_dir, "trace_images.viser"), "wb") as f:
        f.write(replay.recording)
    with open(os.path.join(out_dir, "trace_images_conversion.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    return out
