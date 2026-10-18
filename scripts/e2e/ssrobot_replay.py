# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Replay ssrobot runs of Geodude and ADA from their run bundles (#4).

Needs ssrobot with its MuJoCo extra, the geodude_assets and ada_assets packages, and
ssrobot's reference scenes (``references/scenes`` in the ssrobot repository).

Usage: python -I scripts/e2e/ssrobot_replay.py SCENES_DIR OUT_DIR

For each robot, one deterministic run in its reference scene: the manipulator's joints
move 0.15 rad and settle, a snapshot every 10 ticks, every non-image channel observed at the start
and then every 50 ticks, and the cylinder declared held and then released partway. The run is
bundled with ``ssrobot.write_bundle``. Then, with MuJoCo runtimes and robot contexts made
unusable, ``mj_viser.replay`` converts the bundle; the documented command converts a
moved copy. The checks, on the recording itself:

- every frame puts every body and geom where the runtime had it at that snapshot;
- each command, status, scene revision, and observation is in the label at its
  recorded time, and the label ends showing exactly what the same channel model, fed
  live as the run's sink, showed;
- every declared channel is displayed (with readings), unavailable, or unsupported;
- both conversions give byte-identical recordings and reports;
- a bundle with a missing or a changed scene file is refused, and nothing is written;
- an output directory inside the bundle, directly or through a symlink, is refused, and
  the bundle stays byte-identical and verifiable.

Writes, per robot, the bundle, recording.viser and conversion.json, and
ssrobot_replay.json for both robots, to OUT_DIR.
"""

from __future__ import annotations

import copy
import faulthandler
import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import ssrobot  # noqa: E402
import ssrobot.mujoco  # noqa: E402
from ssrobot import (  # noqa: E402
    JointTrajectory,
    JsonlTrace,
    ObservationRequest,
    RobotContext,
    load_installed_package,
    write_bundle,
)
from ssrobot.mujoco import MujocoRuntime, render_handle  # noqa: E402

import mj_viser  # noqa: E402
from e2e.oracles import check_world_poses, recorded_local_poses, replay_recording  # noqa: E402
from mj_viser import (
    ChannelModel,  # noqa: E402
    replay,  # noqa: E402
)

faulthandler.dump_traceback_later(600, exit=True)

ROBOTS = {"geodude": ("geodude_assets", "right_arm"), "ada": ("ada_assets", "arm")}
DELTA = 0.15  # rad, every joint of the manipulator
DURATION_NS = 1_000_000_000
SNAPSHOT_EVERY = 10  # ticks
ATTACH_AT, DETACH_AT = 150, 160  # ticks
SETTLE_TICKS = 100
MAX_TICKS = 2_000
OBSERVE_EVERY = 50  # ticks
# Viser sends a node's new pose only when it is not np.allclose (rtol 1e-5) to the last one
# sent, so between nearby frames a body's local pose can lag by about 1e-5 per node.
POSE_ATOL = 1e-4


def run(name: str, scene: Path, out: Path) -> tuple[Path, list[tuple[int, Any, Any]], ChannelModel]:
    """Run, bundle, and return the bundle, per snapshot the runtime's own state, and the
    channel model a live view would have shown, fed as a sink during the run."""
    module, manipulator_name = ROBOTS[name]
    package = load_installed_package(module)
    description = package.description
    manipulator = description.manipulator(manipulator_name)
    group = next(g for g in description.groups if g.name == manipulator.group)
    trace_path = out / "run" / "trace.jsonl"
    trace_path.parent.mkdir(parents=True)
    truth: list[tuple[int, Any, Any]] = []  # (time_ns, model, data) as the runtime had them
    live = ChannelModel(description)
    with (
        JsonlTrace(trace_path) as trace,
        RobotContext(description, runtime := MujocoRuntime(package, scene=scene), sinks=[trace, live]) as ctx,
    ):
        handle = render_handle(ctx)
        # Every channel the runtime provides, except images (not shown yet).
        quantity = {c.name: str(c.quantity) for c in description.channels}
        request = ObservationRequest(
            channels=tuple(c for c in ctx.info.channels if quantity[c] not in ("rgb_image", "depth_image"))
        )

        def snapshot() -> Any:
            taken = ctx.snapshot()
            frame = handle.capture()  # the oracle: the runtime's state, through its read-only handle
            truth.append((frame.time.time_ns, handle.model, copy.deepcopy(handle.data)))
            return taken

        start = snapshot()
        ctx.observe(request)
        positions = dict(zip(start.joints, start.positions, strict=True))
        first = tuple(positions[j] for j in group.joints)
        execution = ctx.submit(
            JointTrajectory(
                group=group.name,
                joints=group.joints,
                time_from_start_ns=(0, DURATION_NS),
                positions=(first, tuple(q + DELTA for q in first)),
            )
        )
        ticks = 0
        while (not execution.done or ticks < SETTLE_TICKS) and ticks < MAX_TICKS:
            ctx.step()
            ticks += 1
            if ticks == ATTACH_AT:
                ctx.attach("cylinder", manipulator.end_effector)
            if ticks == DETACH_AT:
                ctx.detach("cylinder")
            if ticks % SNAPSHOT_EVERY == 0:
                snapshot()
            if ticks % OBSERVE_EVERY == 0:
                ctx.observe(request)
        ctx.observe(request)
        model_name = runtime.mapping.model_name
    bundle = out / "bundle"
    write_bundle(bundle, trace=trace_path, package=package, model=model_name, scene=scene)
    return bundle, truth, live


def _no_runtime(*args: Any, **kwargs: Any) -> None:
    raise SystemExit("the conversion tried to start a robot runtime or context")


def convert_without_runtime(bundle: Path, out: Path) -> dict[str, Any]:
    saved = (MujocoRuntime.open, RobotContext.__init__, mujoco.mj_step, mujoco.mj_forward)
    MujocoRuntime.open = _no_runtime  # type: ignore[method-assign]
    RobotContext.__init__ = _no_runtime  # type: ignore[method-assign]
    mujoco.mj_step = mujoco.mj_forward = _no_runtime
    try:
        return replay.convert_bundle(bundle, out)
    finally:
        MujocoRuntime.open, RobotContext.__init__, mujoco.mj_step, mujoco.mj_forward = saved  # type: ignore[method-assign]


def label_texts(messages: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, str]]:
    """The status label's text over recording time."""
    out = []
    for t, m in messages:
        if m.get("name") != replay.LABEL:
            continue
        text = (m.get("props") or m.get("updates") or {}).get("text")
        if text is not None:
            out.append((t, text))
    return out


def text_at(texts: list[tuple[float, str]], t: float) -> str:
    return [s for when, s in texts if when <= t + 1e-9][-1]


def _tree(root: Path) -> dict[str, str]:
    """Every path under ``root`` and its content hash (directories as "dir")."""
    return {
        str(p.relative_to(root)): "dir" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
    }


def check(name: str, bundle: Path, truth: list[tuple[int, Any, Any]], live: ChannelModel, out: Path) -> dict[str, Any]:
    converted = out / "converted"
    report = convert_without_runtime(bundle, converted)
    recording = (converted / "recording.viser").read_bytes()
    messages = replay_recording(recording)

    # Every frame shows the runtime's state at its snapshot.
    times = [t for t, _, _ in truth]
    if report["frames"]["times_ns"] != times:
        raise SystemExit(f"{name}: frames at {report['frames']['times_ns']}, snapshots at {times}")
    t0 = report["trace"]["start_ns"]
    world = ssrobot.mujoco.materialize_world(ssrobot.open_bundle(bundle))
    scene = mj_viser.MujocoViewer(world.model).scene_report()
    checked = 0
    for i, (t, model, data) in enumerate(truth):
        until = (times[i + 1] if i + 1 < len(times) else report["trace"]["end_ns"] + 1) - t0
        local = recorded_local_poses(messages, (t - t0 + until) / 2e9 if i + 1 < len(times) else float("inf"))
        checked += check_world_poses(local, scene, model, data, f"{name} frame at {t} ns", atol=POSE_ATOL)

    # Each event is in the label at its recorded time.
    records = ssrobot.open_bundle(bundle).records
    texts = label_texts(messages)
    events = []
    submitted = {r.payload.execution: r.payload.command for r in records if r.kind.value == "submitted"}
    for r in records:
        kind = r.kind.value
        if kind == "submitted":
            expect = f"{r.payload.execution}: {type(r.payload.command).__name__} on {r.payload.command.group}"
        elif kind == "status":
            expect = f"{r.payload.execution}: {type(submitted[r.payload.execution]).__name__}"
            expect += f" on {submitted[r.payload.execution].group} · submitted"
            history = [x.payload.state.value for x in records if x.kind.value == "status" and x.sequence <= r.sequence]
            expect += "".join(f" → {state}" for state in history)
        elif kind == "scene":
            expect = f"scene revision {r.payload.revision}"
        elif kind == "observed":
            reading = r.payload.readings[-1]
            expect = f"{reading.channel} = "
            shown = text_at(texts, (r.time_ns - t0) / 1e9)
            line = next((x for x in shown.split("\n") if x.startswith(expect)), "")
            if f"{reading.value[0]:.3f}" not in line or f"(t = {reading.stamp.time_ns / 1e9:.3f} s)" not in line:
                raise SystemExit(f"{name}: {reading.channel} at {r.time_ns} ns shows {line!r}")
        else:
            continue
        shown = text_at(texts, (r.time_ns - t0) / 1e9)
        if expect not in shown:
            raise SystemExit(f"{name}: {kind} at {r.time_ns} ns not shown; the label reads {shown!r}")
        events.append(kind)

    # Every declared channel is accounted for, and the replay ends showing exactly what
    # the same model, fed live as the run's sink, showed.
    opened = ssrobot.open_bundle(bundle)
    confirmed = set(next(r.payload.channels for r in opened.records if r.kind.value == "opened"))
    coverage = report["channels"]
    for c in opened.package.description.channels:
        quantity = str(c.quantity)
        want = (
            "unsupported"
            if quantity in ("rgb_image", "depth_image")
            else "displayed"
            if c.name in confirmed
            else "unavailable"
        )
        got = coverage.get(c.name, {})
        if got.get("status") != want or (want == "displayed" and not got.get("readings")):
            raise SystemExit(f"{name}: channel {c.name} is {got}, expected {want} with readings")
    if coverage != live.coverage():
        raise SystemExit(f"{name}: replay coverage differs from live")
    final = texts[-1][1]
    if final != "\n".join(live.lines()):
        raise SystemExit(f"{name}: replay ends showing {final!r}; live showed {live.lines()}")

    # The documented command, on a moved copy, gives the same recording and report.
    moved = out / "moved" / "bundle"
    moved.parent.mkdir()
    shutil.move(str(bundle), moved)
    again = out / "converted-moved"
    subprocess.run([sys.executable, "-I", "-m", "mj_viser.replay", str(moved), str(again)], check=True)
    shutil.move(str(moved), bundle)
    same = {f: (converted / f).read_bytes() == (again / f).read_bytes() for f in ("recording.viser", "conversion.json")}
    if not all(same.values()):
        raise SystemExit(f"{name}: a moved bundle converted differently: {same}")
    shutil.rmtree(again)
    shutil.rmtree(moved.parent)

    # An output inside the bundle, directly or through a symlink, is refused; the bundle
    # stays byte-identical and still verifies.
    before = _tree(bundle)
    link = out / "bundle-link"
    link.symlink_to(bundle.resolve(), target_is_directory=True)
    inside = {}
    for case, dest in (("inside", bundle / "replay"), ("through_symlink", link / "replay")):
        result = subprocess.run(
            [sys.executable, "-I", "-m", "mj_viser.replay", str(bundle), str(dest)], capture_output=True, text=True
        )
        if result.returncode != 2 or "inside the bundle" not in result.stderr:
            raise SystemExit(f"{name}: an output {case} the bundle was not refused: {result}")
        if _tree(bundle) != before:
            raise SystemExit(f"{name}: refusing an output {case} the bundle changed the bundle")
        ssrobot.open_bundle(bundle)
        inside[case] = "refused; bundle unchanged and verifies"
    link.unlink()

    # A broken scene closure is refused before anything is written.
    scene_file = next(p for p in (bundle / "scene").rglob("*.xml"))
    refusals = {}
    for case in ("missing", "changed"):
        broken = out / f"broken-{case}"
        shutil.copytree(bundle, broken)
        target = broken / scene_file.relative_to(bundle)
        if case == "missing":
            target.unlink()
        else:
            target.write_text(target.read_text().replace("rgba=", "rgba= "))
        dest = out / f"converted-{case}"
        result = subprocess.run(
            [sys.executable, "-I", "-m", "mj_viser.replay", str(broken), str(dest)], capture_output=True, text=True
        )
        if result.returncode == 0 or dest.exists() or any(out.glob(f".converted-{case}.*")):
            raise SystemExit(f"{name}: a bundle with a {case} scene file was converted: {result}")
        refusals[case] = result.stderr.strip().split(":")[1].strip()
        shutil.rmtree(broken)

    return {
        "world": report["world"],
        "frames": report["frames"]["count"],
        "duration_s": report["recording"]["duration_s"],
        "poses_checked": checked,
        "events_checked": {k: events.count(k) for k in sorted(set(events))},
        "not_shown": report["not_shown"],
        "intervals_without_frame": len(report["intervals_without_frame"]),
        "moved_bundle_converts_identically": True,
        "refused": refusals,
        "channels": coverage,
        "replay_shows_what_live_showed": True,
        "output_inside_bundle": inside,
        "recording_sha256": hashlib.sha256(recording).hexdigest(),
        "conversion_sha256": hashlib.sha256((converted / "conversion.json").read_bytes()).hexdigest(),
    }


def main(scenes: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    robots = {}
    for name in ROBOTS:
        where = out / name
        bundle, truth, live = run(name, Path(scenes) / f"{name}-cylinder.xml", where)
        robots[name] = check(name, bundle, truth, live, where)
    report = {
        "robots": robots,
        "versions": {p: version(p) for p in ("mj-viser", "ssrobot", "mujoco", "viser")},
    }
    (out / "ssrobot_replay.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
