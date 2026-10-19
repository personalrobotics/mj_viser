# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Joint-motion previews, per browser, without ssrobot.

Motions of the fixture model's hinge and slide: one with a three-waypoint path and a
1.234 s timed motion, one with a highlighted state whose caption carries markup. The
checks: a motion with nothing to show, of the wrong width, or of joints the model lacks
is refused, as is a viewer of another model; with two browsers, the one that scrubs a
motion to its last waypoint gets a ghost at MuJoCo's kinematics of it and the other gets
none; the timed motion's last tick is its exact end; the highlight shows at its
configuration with its caption escaped; a stale motion's ghost hides; and a recorded
motion shows every waypoint and every sample at its own time, nothing after the last.
"""

from __future__ import annotations

import time
from importlib.metadata import version
from typing import Any

import msgspec
import mujoco
import numpy as np
from websockets.sync.client import connect

from e2e.channels import CAMERA, _frames
from e2e.fixtures import expect_raises, kinematics
from e2e.oracles import replay_recording
from mj_viser import Highlight, JointMotion, MotionPreview, MujocoViewer
from mj_viser.preview import record_motion

WAYPOINTS = ((0.0, 0.0), (0.3, 0.1), (0.6, 0.2))
DURATION = 1.234  # 10 ms ticks do not divide it; the last must still be the end


def _at(t: float) -> tuple[float, float]:
    s = min(max(t / DURATION, 0.0), 1.0)
    return 0.6 * s, 0.2 * s


def _samples(period_ns: int) -> list[tuple[int, tuple[float, float]]]:
    end = round(DURATION * 1e9)
    return [(t, _at(t / 1e9)) for t in [*range(0, end, period_ns), end]]


def _state(configuration: Any, data: mujoco.MjData) -> None:
    data.qpos[:2] = configuration


def _motion(**changes: Any) -> JointMotion:
    fields: dict[str, Any] = {
        "joints": ("hinge", "slide"),
        "state": _state,
        "waypoints": WAYPOINTS,
        "duration_s": DURATION,
        "at": _at,
        "samples": _samples,
    }
    return JointMotion(**{**fields, **changes})


def _ghost(frames: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out = {}
    for m in frames:
        name = m.get("name", "")
        if m["type"] == "SetPositionMessage" and name.startswith("/motion_preview/b") and name.count("/") == 2:
            out[name] = np.asarray(m["position"], float)
    return out


def _choose(ws: Any, frames: list[dict[str, Any]], label: str, value: Any) -> None:
    widget = [m for m in frames if m["type"].startswith("Gui") and m.get("props", {}).get("label") == label][-1]
    ws.send(msgspec.msgpack.encode({"type": "GuiUpdateMessage", "uuid": widget["uuid"], "updates": {"value": value}}))


def check_preview(model: mujoco.MjModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    previews = MotionPreview(model)
    out["refused"] = {
        "nothing_to_show": expect_raises(ValueError, lambda: _motion(waypoints=None, at=None, duration_s=None)),
        "wrong_width": expect_raises(ValueError, lambda: _motion(waypoints=((0.0,),))),
        "unknown_joint": expect_raises(ValueError, lambda: previews.add("x", _motion(joints=("hinge", "elbow")))),
    }
    out["refused"].update(
        {
            "empty_path_alone": expect_raises(ValueError, lambda: _motion(waypoints=(), at=None, duration_s=None)),
            "negative_duration": expect_raises(ValueError, lambda: _motion(duration_s=-1.0)),
            "nan_duration": expect_raises(ValueError, lambda: _motion(duration_s=float("nan"))),
            "nan_waypoint": expect_raises(ValueError, lambda: _motion(waypoints=((0.0, float("nan")),))),
            "reserved_name": expect_raises(ValueError, lambda: previews.add("(none)", _motion())),
            "highlight_named_path": expect_raises(ValueError, lambda: Highlight((0.0, 0.0), (), label="path")),
            "timed_empty_configuration": expect_raises(
                ValueError, lambda: previews.ghost_state(_motion(at=lambda t: ()), ())
            ),
            "timed_nan_configuration": expect_raises(
                ValueError, lambda: previews.ghost_state(_motion(), (float("nan"), 0.0))
            ),
            "zero_period": expect_raises(ValueError, lambda: record_motion(model, _motion(), period_ns=0)),
            "negative_waypoint_spacing": expect_raises(
                ValueError, lambda: record_motion(model, _motion(), waypoint_s=-1)
            ),
            "descending_samples": expect_raises(
                ValueError,
                lambda: record_motion(model, _motion(samples=lambda p: [(0, (0, 0)), (2, (0, 0)), (1, (0, 0))])),
            ),
            "negative_sample": expect_raises(
                ValueError, lambda: record_motion(model, _motion(samples=lambda p: [(-1, (0, 0))]))
            ),
        }
    )
    if _motion(waypoints=()).waypoints is not None:
        raise SystemExit("an empty path was kept as a path")
    with MujocoViewer(mujoco.MjModel.from_xml_string("<mujoco/>"), port=0) as other:
        out["refused"]["viewer_of_another_model"] = expect_raises(ValueError, lambda: other.add_panel(previews))
    highlight = Highlight(configuration=(0.1, 0.05), lines=("collision of slider and <world>",))
    previews.add("plan", _motion())
    previews.add("bad", _motion(highlight=highlight))

    slider = model.body("slider").id
    moving = {f"/motion_preview/b{model.body(b).id}" for b in ("arm", "slider", "cap")}
    with MujocoViewer(model, port=0) as viewer:
        viewer.add_panel(previews)
        url = viewer.url.replace("http://", "ws://")
        clients = [connect(url, subprotocols=[f"viser-v{version('viser')}"], max_size=None) for _ in range(2)]
        try:
            for ws in clients:
                ws.send(msgspec.msgpack.encode(CAMERA))
            seen = [_frames(ws, 1.0) for ws in clients]
            _choose(clients[0], seen[0], "Motion", "plan")
            time.sleep(0.3)
            seen[0] += _frames(clients[0], 0.5)
            _choose(clients[0], seen[0], "Sample", 2)
            first = _frames(clients[0], 1.0)
            other = _frames(clients[1], 0.5)
            ghost = _ghost(seen[0] + first)
            if set(ghost) != moving or _ghost(seen[1] + other):
                raise SystemExit(f"ghost bodies {sorted(ghost)} for the chooser, {sorted(_ghost(other))} for the other")
            for name in moving:
                b = int(name.rsplit("b", 1)[1])
                if not np.allclose(ghost[name], kinematics(model, [0.6, 0.2]).xpos[b], atol=1e-9):
                    raise SystemExit(f"ghost {name} at {ghost[name]}")
            out["ghost_at_last_waypoint"] = sorted(moving)

            _choose(clients[0], seen[0] + first, "Show", "trajectory")
            time.sleep(0.3)
            first += _frames(clients[0], 0.5)
            _choose(clients[0], seen[0] + first, "Sample", 10**6)
            first += _frames(clients[0], 1.0)
            got = _ghost(first)[f"/motion_preview/b{slider}"]
            if not np.allclose(got, kinematics(model, [0.6, 0.2]).xpos[slider], atol=1e-9):
                raise SystemExit(f"the timed motion's last tick shows {got}, not its end")
            out["timed_end"] = "exact"

            _choose(clients[0], seen[0] + first, "Motion", "bad")
            time.sleep(0.3)
            more = _frames(clients[0], 0.5)
            _choose(clients[0], seen[0] + first + more, "Show", "highlighted state")
            shown = _frames(clients[0], 1.0)
            captions = [
                (m.get("updates") or m.get("props") or {}).get("content")
                for m in more + shown
                if m["type"] in ("GuiUpdateMessage", "GuiHtmlMessage")
            ]
            caption = [c for c in captions if c and "collision" in c]
            if not caption or "&lt;world&gt;" not in caption[-1]:
                raise SystemExit(f"highlight caption {caption[-1:]}")
            if not np.allclose(
                _ghost(shown)[f"/motion_preview/b{slider}"], kinematics(model, [0.1, 0.05]).xpos[slider]
            ):
                raise SystemExit("the highlight's ghost is not at its configuration")
            out["highlight"] = caption[-1]

            previews.set_stale("bad", "the scene changed")
            hidden = _frames(clients[0], 1.0)
            if not any(m["type"] == "SetSceneNodeVisibilityMessage" and not m["visible"] for m in hidden):
                raise SystemExit("a stale motion's ghost stayed visible")
            out["stale"] = previews.stale("bad")
        finally:
            for ws in clients:
                ws.close()

    # Recorded: waypoints 0.5 s apart, then 0.5 s later every sample at its own time,
    # including the irregular last interval, and nothing after.
    recording, report = record_motion(model, _motion(), period_ns=250_000_000)
    samples = [0, 250_000_000, 500_000_000, 750_000_000, 1_000_000_000, 1_234_000_000]
    if [p["index"] for p in report["path"]] != [0, 1, 2] or [s["time_ns"] for s in report["trajectory"]] != samples:
        raise SystemExit(f"recorded samples {report}")
    positions = [
        (round(t, 6), np.asarray(m["position"]))
        for t, m in replay_recording(recording)
        if m["type"] == "SetPositionMessage" and m.get("name") == f"/motion_preview/b{slider}"
    ]
    expected_at = {0.5: [0.3, 0.1], 1.0: [0.6, 0.2]}
    expected_at.update({round(1.5 + t / 1e9, 6): list(_at(t / 1e9)) for t in samples[1:]})
    for when, q in expected_at.items():
        got = [p for at, p in positions if at <= when + 1e-9][-1]
        if not np.allclose(got, kinematics(model, q).xpos[slider], atol=1e-9):
            raise SystemExit(f"recorded ghost at {when} s is {got}")
    if abs(max(at for at, _ in positions) - 2.734) > 1e-6:
        raise SystemExit("the recording runs past its last sample")
    out["recorded"] = {"path": len(report["path"]), "trajectory": len(report["trajectory"])}

    # A timed motion whose first sample is later than its start keeps that offset.
    late = _motion(waypoints=None, samples=lambda p: [(100_000_000, (0.3, 0.1)), (200_000_000, (0.6, 0.2))])
    recording, report = record_motion(model, late)
    first = min(t for t, m in replay_recording(recording) if m.get("name") == f"/motion_preview/b{slider}")
    if [round(s["recorded_s"], 6) for s in report["trajectory"]] != [0.1, 0.2] or round(first, 6) != 0.1:
        raise SystemExit(f"a late first sample was recorded at {first} s: {report['trajectory']}")
    out["first_sample_offset_kept"] = 0.1

    # A carried body moves with its whole subtree.
    held = mujoco.MjModel.from_xml_string(CARRIED)
    bodies = MotionPreview(held).ghost_bodies(
        JointMotion(joints=("hinge",), state=lambda q, d: None, waypoints=((0.0,),), carried=("held",))
    )
    names = sorted(held.body(b).name for b in bodies)
    if names != ["arm", "held", "held_child"]:
        raise SystemExit(f"carried ghost bodies {names}")
    out["carried_subtree"] = names
    return out


CARRIED = """
<mujoco>
  <worldbody>
    <body name="arm"><joint name="hinge"/><geom size="0.05"/></body>
    <body name="held" pos="1 0 0"><freejoint/><geom size="0.05"/>
      <body name="held_child" pos="0 0 0.1"><geom size="0.02"/></body>
    </body>
  </worldbody>
</mujoco>
"""
