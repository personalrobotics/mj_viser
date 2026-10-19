# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Live ssrobot view on Geodude and ADA, with proof that viewing changes nothing (#28).

Needs ssrobot with its MuJoCo extra and the geodude_assets and ada_assets packages.

Usage: python -I scripts/e2e/ssrobot_live.py OUT_DIR

For each robot, one deterministic motion (its manipulator's joints moved 0.15 rad by a
two-point trajectory, then settled, every channel but depth observed every 50 ticks) runs
twice under a JSONL trace:

- without a viewer;
- with a LiveView refreshed every tick, with channel panels fed by a ChannelModel sink,
  which meanwhile vandalizes the data it holds.
  A browser connects and never reads; at tick 100 it disconnects while the view stays
  open and keeps rendering; at tick 200 the view detaches and a new view attaches with a
  new render handle.

The two traces must be byte-identical. The viewed run also records a .viser file and
checks each rendered frame against the runtime's own joint positions. Afterwards a view
built for another scene is refused, and a view whose runtime reopened goes stale instead
of raising. Writes ssrobot_live.json, the traces, and the recordings to OUT_DIR.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import msgspec  # noqa: E402
import numpy as np  # noqa: E402
from ssrobot import (  # noqa: E402
    JointTrajectory,
    JsonlTrace,
    ObservationRequest,
    RobotContext,
    load_installed_package,
)
from ssrobot.mujoco import MujocoRuntime  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from mj_viser import ChannelModel, LiveView  # noqa: E402
from mj_viser.integrations.ssrobot import render_source  # noqa: E402

faulthandler.dump_traceback_later(600, exit=True)

ROBOTS = {"geodude": ("geodude_assets", "right_arm"), "ada": ("ada_assets", "arm")}
DELTA = 0.15  # rad, every joint of the manipulator
DURATION_NS = 1_000_000_000
SETTLE_TICKS = 100
MAX_TICKS = 2_000
OBSERVE_EVERY = 50  # ticks; both runs observe every channel the runtime provides, except depth


def _vandalize(view: LiveView) -> None:
    """Everything a careless viewer could do to the data it holds."""
    for data in (view._source.data, view.viewer.data):  # type: ignore[attr-defined]
        data.qpos[:] = 3.0
        data.qvel[:] = 3.0


def _silent_client(url: str) -> Any:
    """A browser that connects and then never reads: the slowest possible client."""
    # No message size cap: a browser has none, and a robot scene is several MB.
    ws = connect(url.replace("http://", "ws://"), subprotocols=[f"viser-v{version('viser')}"], max_size=None)
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
    ws.send(msgspec.msgpack.encode(camera))
    return ws


def _joint_positions(ctx: RobotContext, names: tuple[str, ...]) -> list[float]:
    snapshot = ctx.snapshot()
    by_name = dict(zip(snapshot.joints, snapshot.positions))
    return [float(by_name[n]) for n in names]


def _qpos(model: Any, data: Any, joints: tuple[str, ...]) -> list[float]:
    return [float(data.qpos[model.jnt_qposadr[model.joint(n).id]]) for n in joints]


def _check_frame(view: LiveView, runtime: MujocoRuntime, joints: tuple[str, ...]) -> None:
    """The rendered state is the runtime's own.

    The oracle reads the runtime's simulation state directly (test-only): asking the
    context for a snapshot would itself add trace records and break the comparison.
    """
    shown = _qpos(view.viewer.model, view.viewer.data, joints)
    truth = _qpos(runtime._model, runtime._data, joints)  # noqa: SLF001
    if not np.allclose(shown, truth, atol=1e-9):
        raise SystemExit(f"rendered joints {shown} differ from the runtime's {truth}")


def _check_channels(channels: ChannelModel, observation: Any) -> None:
    """The live channel model holds exactly what the runtime just observed."""
    for reading in observation.readings:
        if hasattr(reading.value, "shape"):  # an image: the panel's image is the runtime's
            image, stamp = channels.image(reading.channel)
            expected = np.frombuffer(reading.value.data, np.uint8).reshape(reading.value.shape)
            if image is None or stamp != reading.stamp.time_ns or not np.array_equal(image, expected):
                raise SystemExit(f"image channel {reading.channel} does not show the observed image")
            continue
        _, history = channels.history(reading.channel)
        if not history or history[-1] != (reading.stamp.time_ns, tuple(float(v) for v in reading.value)):
            raise SystemExit(f"channel {reading.channel} shows {history[-1:]}, observed {reading}")


def run(name: str, out: Path, viewed: bool) -> dict[str, Any]:
    module, manipulator = ROBOTS[name]
    package = load_installed_package(module)
    description = package.description
    group = next(g for g in description.groups if g.name == description.manipulator(manipulator).group)
    mode = "viewed" if viewed else "plain"
    result: dict[str, Any] = {}
    channels = ChannelModel(description)  # the viewed run's channel panels read this sink
    with (
        JsonlTrace(out / f"{name}-{mode}.trace.jsonl") as trace,
        RobotContext(
            description, runtime := MujocoRuntime(package), sinks=[trace, channels] if viewed else [trace]
        ) as ctx,
    ):
        quantity = {c.name: str(c.quantity) for c in description.channels}
        request = ObservationRequest(channels=tuple(c for c in ctx.info.channels if quantity[c] != "depth_image"))
        view = None
        if viewed:
            handle = render_source(ctx)
            view = LiveView(
                handle, expected_world=handle.world, channels=channels, max_rate_hz=None, port=0, label=name
            )
            view.open()
            recorder = view.viewer.server.get_scene_serializer()
            silent = _silent_client(view.viewer.url)
            result["world"] = view.world
        start = _joint_positions(ctx, group.joints)
        goal = tuple(q + DELTA for q in start)
        execution = ctx.submit(
            JointTrajectory(
                group=group.name,
                joints=group.joints,
                time_from_start_ns=(0, DURATION_NS),
                positions=(tuple(start), goal),
            )
        )
        ticks, frames, detached = 0, 0, False
        while (not execution.done or ticks < SETTLE_TICKS) and ticks < MAX_TICKS:
            ctx.step()
            ticks += 1
            if ticks % OBSERVE_EVERY == 0:
                observation = ctx.observe(request)
                if viewed:
                    _check_channels(channels, observation)
            if view is not None:
                if view.refresh() is not None:
                    frames += 1
                    _check_frame(view, runtime, group.joints)
                    if recorder is not None:
                        recorder.insert_sleep(0.002)
                _vandalize(view)
                if ticks == 100:
                    # The browser disconnects; the same view stays open and keeps rendering.
                    silent.close()
                    frames_at_disconnect = frames
                if not detached and ticks == 200:
                    if frames - frames_at_disconnect < 100:
                        raise SystemExit("the view stopped rendering after the browser disconnected")
                    result["frames_after_browser_disconnect"] = frames - frames_at_disconnect
                    # Detach and attach a new view on a new handle; the runtime keeps going.
                    recording = recorder.serialize()  # the first attachment's recording
                    recorder = None
                    view.close()
                    view = LiveView(render_source(ctx), channels=channels, max_rate_hz=None, port=0, label=name)
                    view.open()
                    detached = True
        result.update(status=execution.status.state.value, ticks=ticks, end=_joint_positions(ctx, group.joints))
        if view is not None:
            result.update(frames_rendered=frames, reattached=detached, channels=channels.coverage())
            view.close()
            (out / f"{name}.viser").write_bytes(recording)
            result["recording_sha256"] = hashlib.sha256(recording).hexdigest()
            stale_view = LiveView(render_source(ctx), max_rate_hz=None, port=0)
    if viewed:
        # The runtime reopens: the old handle no longer describes it, so the view goes stale.
        with RobotContext(description, runtime):  # the same runtime, reopened
            stale_view.open()
            refreshed = stale_view.refresh(force=True)
            result["after_reopen"] = {"rendered": refreshed is not None, "stale": stale_view.stale}
            stale_view.close()
    return result


def robot(name: str, out: Path) -> dict[str, Any]:
    plain, viewed = run(name, out, viewed=False), run(name, out, viewed=True)
    identical = (out / f"{name}-plain.trace.jsonl").read_bytes() == (out / f"{name}-viewed.trace.jsonl").read_bytes()
    if not identical:
        raise SystemExit(f"{name}: the trace changed when a viewer was attached")
    if plain["end"] != viewed["end"] or plain["status"] != viewed["status"]:
        raise SystemExit(f"{name}: the motion's outcome changed when a viewer was attached")
    if viewed["after_reopen"] != {"rendered": False, "stale": "stale_render"}:
        raise SystemExit(f"{name}: a view on a reopened runtime did {viewed['after_reopen']}")
    package = load_installed_package(ROBOTS[name][0])
    with RobotContext(package.description, MujocoRuntime(package)) as ctx:
        try:
            LiveView(render_source(ctx), expected_world="another scene")
        except ValueError as e:
            refused = str(e)
        else:
            raise SystemExit(f"{name}: a view for another scene was accepted")
    trace = out / f"{name}-plain.trace.jsonl"
    return {
        "world": viewed["world"],
        "status": plain["status"],
        "ticks": plain["ticks"],
        "frames_rendered": viewed["frames_rendered"],
        "reattached_without_restart": viewed["reattached"],
        "frames_after_browser_disconnect": viewed["frames_after_browser_disconnect"],
        "channels": viewed["channels"],
        "traces_identical_with_and_without_viewer": identical,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "recording_sha256": viewed["recording_sha256"],
        "after_runtime_reopen": viewed["after_reopen"],
        "other_scene_refused": refused.replace(viewed["world"], "<world>"),
    }


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "robots": {name: robot(name, out) for name in ROBOTS},
        "versions": {p: version(p) for p in ("mj-viser", "ssrobot", "mujoco", "viser")},
    }
    (out / "ssrobot_live.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
