# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Preview Geodude and ADA plans as ghosts, with proof that previewing changes nothing (#3).

Needs ssrobot with its MuJoCo and planning extras, the geodude_assets and ada_assets
packages, and ssrobot's reference scenes.

Usage: python -I scripts/e2e/ssrobot_preview.py SCENES_DIR OUT_DIR

For each robot in its reference scene, from one snapshot, two plans of different goal
kinds: "joint" (the arm's joints moved 0.2 rad) and "offset" (the end effector 5 cm
straight up). The run happens twice under a JSONL trace, without and with previews:

- a LiveView with a PlanPreview of both results, and two browsers: one scrubs "joint"
  to its last path waypoint, the other "offset" to the end of its trajectory. Each
  browser's ghost puts the tool frame where ssrobot's planning scene puts it for that
  sample, and neither browser receives the other's ghost;
- after the scene changes (the cylinder declared held), the previews go stale and the
  results are refused; a lift planned with the cylinder held is previewed, and the
  ghost cylinder rides the hand exactly as ssrobot's planning scene moves it;
- the recorded preview shows each trajectory sample at its own time.

The two traces must be byte-identical. Writes, per robot, the results, the preview
recording of "offset" (record_preview) and its report, and ssrobot_preview.json.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import msgspec  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from ssrobot import JsonlTrace, RobotContext, dumps, load_installed_package  # noqa: E402
from ssrobot.mujoco import MujocoRuntime, materialize_world_from  # noqa: E402
from ssrobot.planning import CBiRRT, SsikKinematics  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from e2e.channels import CAMERA, _frames  # noqa: E402
from e2e.oracles import replay_recording  # noqa: E402
from mj_viser import LiveView  # noqa: E402
from mj_viser.integrations.ssrobot import PlanPreview, record_preview, render_source  # noqa: E402
from mj_viser.recording import TEXT  # noqa: E402

faulthandler.dump_traceback_later(900, exit=True)

ROBOTS = {"geodude": ("geodude_assets", "right_arm"), "ada": ("ada_assets", "arm")}
DELTA = 0.2  # rad
LIFT = 0.05  # m


def _plans(ctx: Any, package: Any, manipulator: str) -> tuple[Any, dict[str, Any]]:
    arm = ctx.manipulator(manipulator)
    planner = CBiRRT(package.description, kinematics=SsikKinematics.from_package(package, arm.name))
    snapshot = ctx.snapshot()
    group = package.description.group(package.description.manipulator(manipulator).group)
    start = tuple(snapshot.position(j) for j in group.joints)
    results = {
        "joint": arm.plan_to_configuration(tuple(q + DELTA for q in start), snapshot=snapshot, planner=planner),
        "offset": arm.plan_to_end_effector_offset((0.0, 0.0, 1.0), LIFT, snapshot=snapshot, planner=planner),
    }
    for name, result in results.items():
        if not result.succeeded:
            raise SystemExit(f"{name} plan {result.status.value}: {result.diagnostics}")
    return snapshot, results


def _held_plan(ctx: Any, package: Any, manipulator: str) -> tuple[Any, Any]:
    """With the cylinder declared held: a snapshot and a 5 cm lift planned in it."""
    arm = ctx.manipulator(manipulator)
    planner = CBiRRT(package.description, kinematics=SsikKinematics.from_package(package, arm.name))
    snapshot = ctx.snapshot()
    result = arm.plan_to_end_effector_offset((0.0, 0.0, 1.0), LIFT, snapshot=snapshot, planner=planner)
    if not result.succeeded:
        raise SystemExit(f"held lift plan {result.status.value}: {result.diagnostics}")
    return snapshot, result


def _matrix(pose: Any) -> np.ndarray:
    out = np.eye(4)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(pose.quat_wxyz, float))
    out[:3, :3] = rotation.reshape(3, 3)
    out[:3, 3] = pose.position
    return out


def _choose(ws: Any, frames: list[dict[str, Any]], label: str, value: Any) -> None:
    widget = [m for m in frames if m["type"].startswith("Gui") and m.get("props", {}).get("label") == label][-1]
    ws.send(msgspec.msgpack.encode({"type": "GuiUpdateMessage", "uuid": widget["uuid"], "updates": {"value": value}}))


def _ghost_poses(frames: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses: dict[str, dict[str, Any]] = {}
    for m in frames:
        name = m.get("name", "")
        if not name.startswith("/motion_preview/b") or name.count("/") != 2:
            continue
        if m["type"] == "SetPositionMessage":
            poses.setdefault(name, {})["p"] = np.asarray(m["position"], float)
        elif m["type"] == "SetOrientationMessage":
            poses.setdefault(name, {})["q"] = np.asarray(m["wxyz"], float)
    return {n: (v["p"], v["q"]) for n, v in poses.items() if "p" in v and "q" in v}


def run(name: str, scene: Path, out: Path, preview: bool) -> dict[str, Any]:
    module, manipulator = ROBOTS[name]
    package = load_installed_package(module)
    description = package.description
    mode = "previewed" if preview else "plain"
    result: dict[str, Any] = {}
    with (
        JsonlTrace(out / f"{mode}.trace.jsonl") as trace,
        RobotContext(description, runtime := MujocoRuntime(package, scene=scene), sinks=[trace]) as ctx,
    ):
        snapshot, results = _plans(ctx, package, manipulator)
        if not preview:
            ctx.attach("cylinder", description.manipulator(manipulator).end_effector)  # as the previewed run does
            _held_plan(ctx, package, manipulator)
            return result
        for key, r in results.items():
            (out / f"{key}.result.json").write_text(dumps(r) + "\n")
        world = materialize_world_from(package, world=runtime.world(), scene=scene)
        view = LiveView(render_source(ctx), max_rate_hz=None, port=0, label=name)
        previews = PlanPreview(world, description, live=view)
        view.viewer.add_panel(previews)
        url = view.open().replace("http://", "ws://")
        for key, r in results.items():
            previews.add_result(key, r, snapshot)
        frame_name = description.manipulator(manipulator).tool_frame
        body = world.model.body(frame_name).id  # the tool frame is a body in both packages
        clients = [connect(url, subprotocols=[f"viser-v{version('viser')}"], max_size=None) for _ in range(2)]
        try:
            for ws in clients:
                ws.send(msgspec.msgpack.encode(CAMERA))
            seen = [_frames(ws, 1.5) for ws in clients]
            choices = {0: ("joint", "path"), 1: ("offset", "trajectory")}
            for i, (key, show) in choices.items():
                _choose(clients[i], seen[i], "Motion", key)
                time.sleep(0.3)
                seen[i] += _frames(clients[i], 0.5)
                _choose(clients[i], seen[i], "Show", show)
                time.sleep(0.3)
                seen[i] += _frames(clients[i], 0.5)
                _choose(clients[i], seen[i], "Sample", 10**6)  # clamped to the last sample
            ghosts = []
            for i, ws in enumerate(clients):
                seen[i] += _frames(ws, 1.5)
                ghosts.append(_ghost_poses(seen[i]))
            # Each browser's tool frame is where the planning scene puts it at that sample.
            scene_for = ctx.planning_scene(snapshot, results["joint"].request.group)
            samples = {
                0: results["joint"].path.waypoints[-1],
                1: results["offset"].trajectory.evaluate(results["offset"].trajectory.duration)[0],
            }
            checked = {}
            for i, q in samples.items():
                expected = scene_for.forward_kinematics(tuple(q), frame_name)
                position, quat = ghosts[i][f"/motion_preview/b{body}"]
                gap = float(np.abs(position - np.asarray(expected.position)).max())
                turn = float(min(np.abs(quat - expected.quat_wxyz).max(), np.abs(quat + expected.quat_wxyz).max()))
                if gap > 1e-6 or turn > 1e-6:
                    raise SystemExit(f"{name}: browser {i}'s ghost tool frame is off by {gap} m, {turn}")
                checked[choices[i][0]] = {"position_m": gap, "quaternion": turn}
            tool = f"/motion_preview/b{body}"
            if np.allclose(ghosts[0][tool][0], ghosts[1][tool][0]):
                raise SystemExit(f"{name}: both browsers show the same ghost")
            result["ghost_tool_frame_error"] = checked

            # The scene changes: previews go stale, and stale results are refused.
            ctx.attach("cylinder", description.manipulator(manipulator).end_effector)
            view.refresh(force=True)
            result["stale_after_scene_change"] = {k: previews.stale(k) for k in results}
            try:
                previews.add_result("joint again", results["joint"], snapshot)
            except ValueError as e:
                result["stale_result_refused"] = str(e)
            else:
                raise SystemExit(f"{name}: a result from an earlier scene revision was accepted")

            # Held: the cylinder, declared held, rides the hand in the ghost exactly as the
            # planning scene moves it.
            held_snapshot, held = _held_plan(ctx, package, manipulator)
            previews.add_result("held", held, held_snapshot)
            time.sleep(0.3)
            seen[0] += _frames(clients[0], 0.5)
            _choose(clients[0], seen[0], "Motion", "held")
            time.sleep(0.3)
            seen[0] += _frames(clients[0], 0.5)
            _choose(clients[0], seen[0], "Show", "trajectory")
            time.sleep(0.3)
            seen[0] += _frames(clients[0], 0.5)
            _choose(clients[0], seen[0], "Sample", 10**6)
            seen[0] += _frames(clients[0], 1.5)
            ghost = _ghost_poses(seen[0])
            attachment = next(a for a in held_snapshot.attachments if a.object == "cylinder")
            effector = description.end_effector(attachment.end_effector).frame
            planning = ctx.planning_scene(held_snapshot, held.request.group)
            hand = _matrix(planning.forward_kinematics(held.trajectory.goal, effector))
            expected = hand @ _matrix(attachment.transform)
            cylinder = world.model.body("cylinder").id
            position, quat = ghost[f"/motion_preview/b{cylinder}"]
            gap = float(np.abs(position - expected[:3, 3]).max())
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, quat)
            turn = float(np.abs(rotation.reshape(3, 3) - expected[:3, :3]).max())
            if gap > 1e-6 or turn > 1e-6:
                raise SystemExit(f"{name}: the held cylinder's ghost is off by {gap} m, {turn}")
            result["held_object_ghost_error"] = {"position_m": gap, "rotation": turn}
        finally:
            for ws in clients:
                ws.close()
            view.close()
        recording, report = record_preview(results["offset"], snapshot, world, description)
        # The recording shows each trajectory sample at its own time, the last too.
        label_times = [
            round(t, 6)
            for t, m in replay_recording(recording)
            if m.get("name") == TEXT and "trajectory" in str((m.get("updates") or {}).get("text", ""))
        ]
        expected_times = [round(s["recorded_s"], 6) for s in report["trajectory"]]
        start = expected_times[0]
        if (
            label_times != expected_times
            or [round(start + s["time_ns"] / 1e9, 6) for s in report["trajectory"]] != expected_times
        ):
            raise SystemExit(f"{name}: trajectory samples recorded at {label_times}, expected {expected_times}")
        intervals = np.diff([s["time_ns"] for s in report["trajectory"]])
        (out / "offset.preview.viser").write_bytes(recording)
        (out / "offset.preview.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        result.update(
            world=world.world,
            snapshot=snapshot.fingerprint(),
            recording_sha256=hashlib.sha256(recording).hexdigest(),
            recorded_samples={"path": len(report["path"]), "trajectory": len(report["trajectory"])},
            recorded_final_interval_ns=int(intervals[-1]),
        )
    return result


def main(scenes: str, out_dir: str) -> None:
    out = Path(out_dir)
    robots = {}
    for name in ROBOTS:
        where = out / name
        where.mkdir(parents=True, exist_ok=True)
        scene = Path(scenes) / f"{name}-cylinder.xml"
        run(name, scene, where, preview=False)
        previewed = run(name, scene, where, preview=True)
        identical = (where / "plain.trace.jsonl").read_bytes() == (where / "previewed.trace.jsonl").read_bytes()
        if not identical:
            raise SystemExit(f"{name}: previewing changed the trace")
        if any(v is None for v in previewed["stale_after_scene_change"].values()):
            raise SystemExit(f"{name}: a preview stayed current after the scene changed")
        robots[name] = {**previewed, "traces_identical_with_and_without_preview": identical}
    report = {
        "robots": robots,
        "versions": {p: version(p) for p in ("ssmjviser", "ssrobot", "mujoco", "viser")},
    }
    (out / "ssrobot_preview.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
