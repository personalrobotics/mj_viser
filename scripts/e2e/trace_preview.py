# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The ssrobot integration's plan previews, on stand-in results (needs ssrobot installed).

Stand-in results over the fixture model's hinge and slide: one with a three-waypoint
path and a one-second trajectory, one with an invalid validation's first invalid state.
The checks: results that do not belong to the snapshot, world, or scene revision, or
that have nothing to show, are refused; with two browsers, the one that scrubs a result
to its last waypoint gets a ghost at MuJoCo's kinematics of that waypoint, and the other
gets no ghost; the first invalid state is shown at its configuration with its location
and contacts; a live scene change makes a preview stale; and a recorded preview steps
through every waypoint and trajectory sample.
"""

from __future__ import annotations

import time
from importlib.metadata import version
from types import SimpleNamespace as NS
from typing import Any

import msgspec
import mujoco
import numpy as np
from websockets.sync.client import connect

from e2e.channels import CAMERA, _frames
from e2e.fixtures import expect_raises, kinematics
from e2e.oracles import replay_recording
from mj_viser import MujocoViewer
from mj_viser.integrations.ssrobot import PlanPreview, record_preview

WAYPOINTS = ((0.0, 0.0), (0.3, 0.1), (0.6, 0.2))


class _Mapping:
    joints = (("hinge", 0), ("slide", 1))
    objects: tuple[tuple[str, int], ...] = ()

    def apply(self, snapshot: Any, data: mujoco.MjData) -> None:
        data.qpos[:2] = snapshot.positions


class _Trajectory:
    duration = 1.234  # the slider's 10 ms ticks do not divide it; the last must still be the end

    def evaluate(self, t: float) -> tuple[tuple[float, ...], None, None]:
        s = min(max(t / self.duration, 0.0), 1.0)
        return (0.6 * s, 0.2 * s), None, None

    def sample(self, period_ns: int) -> NS:
        end = round(self.duration * 1e9)  # as ssrobot samples: the period, then the end
        times = [*range(0, end, period_ns), end]
        return NS(time_from_start_ns=times, positions=[self.evaluate(t / 1e9)[0] for t in times])


def _snapshot(fingerprint: str = "s1", world: str = "w") -> NS:
    return NS(fingerprint=lambda: fingerprint, model=world, description="d", positions=(0.0, 0.0), attachments=())


DESCRIPTION = NS(fingerprint=lambda: "d", end_effectors=())


def _result(snapshot: str = "s1", revision: int = 0, path: bool = True) -> NS:
    request = NS(snapshot=snapshot, revision=revision, description="d", group="arm", joints=("hinge", "slide"))
    return NS(
        request=request,
        path=NS(waypoints=WAYPOINTS) if path else None,
        trajectory=_Trajectory() if path else None,
        validation=(NS(motion="path-1"), NS(motion="trajectory-1")) if path else (),
    )


def _invalid(witness: NS, motion: str = "path-1", group: str = "arm") -> NS:
    validator = NS(snapshot="s1", group=group, joints=("hinge", "slide"))
    return NS(valid=False, witness=witness, motion=motion, validator=validator)


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


def check_trace_preview(model: mujoco.MjModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    world = NS(model=model, world="w", description="d", mapping=_Mapping())
    previews = PlanPreview(world, DESCRIPTION)
    out["refused"] = {
        "other_snapshot": expect_raises(ValueError, lambda: previews.add_result("x", _result("s2"), _snapshot())),
        "other_world": expect_raises(ValueError, lambda: previews.add_result("x", _result(), _snapshot(world="v"))),
        "nothing_to_show": expect_raises(
            ValueError, lambda: previews.add_result("x", _result(path=False), _snapshot())
        ),
        "valid_validation": expect_raises(
            ValueError,
            lambda: previews.add_result("x", _result(), _snapshot(), validation=NS(valid=True, witness=None)),
        ),
    }
    witness = NS(
        failure="collision",
        segment=0,
        fraction=0.3,
        time_s=None,
        configuration=(0.1, 0.05),
        contacts=(NS(first="slider", second="<world>", distance=-0.00123),),
        detail="",
    )
    out["refused"]["validation_of_another_motion"] = expect_raises(
        ValueError,
        lambda: previews.add_result("x", _result(), _snapshot(), validation=_invalid(witness, motion="other")),
    )
    out["refused"]["validation_of_another_group"] = expect_raises(
        ValueError,
        lambda: previews.add_result("x", _result(), _snapshot(), validation=_invalid(witness, group="other")),
    )
    out["refused"]["live_view_of_another_world"] = expect_raises(
        ValueError,
        lambda: PlanPreview(world, DESCRIPTION, live=NS(world="v")),  # type: ignore[arg-type]
    )
    with MujocoViewer(mujoco.MjModel.from_xml_string("<mujoco/>"), port=0) as other:
        out["refused"]["viewer_of_another_model"] = expect_raises(ValueError, lambda: other.add_panel(previews))
    previews.add_result("plan", _result(), _snapshot())
    previews.add_result("bad", _result(), _snapshot(), validation=_invalid(witness))

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
            expected = kinematics(model, [0.6, 0.2])
            for name in moving:
                b = int(name.rsplit("b", 1)[1])
                if not np.allclose(ghost[name], expected.xpos[b], atol=1e-9):
                    raise SystemExit(f"ghost {name} at {ghost[name]}, MuJoCo has {expected.xpos[b]}")
            out["ghost_at_last_waypoint"] = sorted(moving)

            # The trajectory's last tick is its exact end: 1.234 s, not 1.23 s.
            _choose(clients[0], seen[0] + first, "Show", "trajectory")
            time.sleep(0.3)
            first += _frames(clients[0], 0.5)
            _choose(clients[0], seen[0] + first, "Sample", 10**6)
            first += _frames(clients[0], 1.0)
            end = kinematics(model, [0.6, 0.2]).xpos[model.body("slider").id]
            got = _ghost(first)[f"/motion_preview/b{model.body('slider').id}"]
            if not np.allclose(got, end, atol=1e-9):
                raise SystemExit(f"the trajectory's last tick shows {got}, its end is {end}")
            out["trajectory_end"] = "exact"

            _choose(clients[0], seen[0] + first, "Motion", "bad")
            time.sleep(0.3)
            more = _frames(clients[0], 0.5)
            _choose(clients[0], seen[0] + first + more, "Show", "first invalid state")
            shown = _frames(clients[0], 1.0)
            captions = [
                (m.get("updates") or m.get("props") or {}).get("content")
                for m in more + shown
                if m["type"] in ("GuiUpdateMessage", "GuiHtmlMessage")
            ]
            caption = [c for c in captions if c and "first invalid state" in c]
            if not caption or "segment 0 at 0.300" not in caption[-1] or "&lt;world&gt; -1.23 mm" not in caption[-1]:
                raise SystemExit(f"witness caption {caption[-1:]}")
            at = kinematics(model, [0.1, 0.05])
            slider = f"/motion_preview/b{model.body('slider').id}"
            if not np.allclose(_ghost(shown)[slider], at.xpos[model.body("slider").id], atol=1e-9):
                raise SystemExit("the first invalid state's ghost is not at its configuration")
            out["witness"] = caption[-1]
        finally:
            for ws in clients:
                ws.close()

    # Live: a result from another scene revision is refused, and a shown one goes stale.
    with MujocoViewer(model, port=0) as viewer:
        live = NS(frame=NS(revision=0), world="w", viewer=viewer)  # a LiveView's shape
        staling = PlanPreview(world, DESCRIPTION, live=live)  # type: ignore[arg-type]
        staling.add_result("plan", _result(), _snapshot())
        out["refused"]["other_revision"] = expect_raises(
            ValueError, lambda: staling.add_result("old", _result(revision=1), _snapshot())
        )
        viewer.add_panel(staling)
        live.frame = NS(revision=1)
        viewer.update(mujoco.MjData(model))
        out["stale_after_scene_change"] = staling.stale("plan")
    if out["stale_after_scene_change"] is None:
        raise SystemExit("a preview stayed current after the scene changed")

    # A recorded preview: every waypoint 0.5 s apart, then 0.5 s later every trajectory
    # sample at its own time, including the irregular last interval, and nothing after.
    recording, report = record_preview(_result(), _snapshot(), world, DESCRIPTION, period_ns=250_000_000)
    samples = [0, 250_000_000, 500_000_000, 750_000_000, 1_000_000_000, 1_234_000_000]
    if [p["index"] for p in report["path"]] != [0, 1, 2] or [s["time_ns"] for s in report["trajectory"]] != samples:
        raise SystemExit(f"recorded samples {report}")
    slider = model.body("slider").id
    positions = [
        (round(t, 6), np.asarray(m["position"]))
        for t, m in replay_recording(recording)
        if m["type"] == "SetPositionMessage" and m.get("name") == f"/motion_preview/b{slider}"
    ]
    expected_at = {0.5: [0.3, 0.1], 1.0: [0.6, 0.2]}  # waypoints 2 and 3
    expected_at.update({round(1.5 + t / 1e9, 6): list(_Trajectory().evaluate(t / 1e9)[0]) for t in samples[1:]})
    for when, q in expected_at.items():
        got = [p for at, p in positions if at <= when + 1e-9][-1]
        if not np.allclose(got, kinematics(model, q).xpos[slider], atol=1e-9):
            raise SystemExit(f"recorded ghost at {when} s is {got}")
    last = max(at for at, _ in positions)
    if abs(last - 2.734) > 1e-6:
        raise SystemExit(f"the recording's last ghost update is at {last} s, not 2.734 s")
    out["recorded"] = {"path": len(report["path"]), "trajectory": len(report["trajectory"])}
    return out
