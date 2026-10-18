# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Replay a recorded run as a native Viser recording.

The run's trace is the record; a ``.viser`` file made from it is a derived view. Frames
come from the trace's ``SceneSnapshot`` records, at whatever cadence the run took them,
and nothing is interpolated between them. Commands, statuses, scene revisions,
observations, attachment violations, and runtime events are shown in a scene label at
their recorded times. Playback uses Viser's own player and timeline.

``python -m mj_viser.replay BUNDLE OUT_DIR`` converts an ssrobot run bundle: it opens
and verifies the bundle with ``ssrobot.open_bundle``, builds its world with
``ssrobot.mujoco.materialize_world`` (no runtime, context, or physics), and writes
``recording.viser`` and ``conversion.json``. mj_viser does not import ssrobot otherwise;
:func:`record_trace` accepts any records and world of the same shape.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np

from mj_viser.viewer import MujocoViewer

LABEL = "/replay/status"
"""Scene path of the label that shows the trace's events."""

# Shown kinds; ``stepped`` marks time only, and every applied setpoint is counted but not
# shown unless the runtime modified it.
_SHOWN = ("opened", "observed", "compiled", "submitted", "status", "applied", "health", "scene", "violation", "closed")


class SnapshotMapping(Protocol):
    def apply(self, snapshot: Any, data: mujoco.MjData) -> None:
        """Write a snapshot's state into ``data.qpos``; refuse one of another world."""
        ...


class World(Protocol):
    """A recorded world built for viewing, like ssrobot's ``MaterializedWorld``."""

    model: mujoco.MjModel
    world: str
    """The world identity every snapshot must name."""
    description: str
    """The robot description fingerprint every snapshot must name."""
    mapping: SnapshotMapping


class TraceRecord(Protocol):
    """A trace record, like ssrobot's ``TraceRecord``."""

    sequence: int
    kind: Any
    """The record kind; its string value, e.g. ``"snapshot"``, is used."""
    clock: str
    time_ns: int
    payload: Any


@dataclass(frozen=True)
class Replay:
    """A converted run."""

    recording: bytes
    """The ``.viser`` file."""
    report: dict[str, Any]
    """What was shown when, what was not, and every interval without a frame."""


def record_trace(records: Sequence[TraceRecord], world: World, *, label: str | None = None) -> Replay:
    """Convert a trace of ``world`` into a Viser recording.

    Recording time is trace time since the first record. Before the first snapshot the
    world is hidden: there is no recorded state to show.

    Raises:
        ValueError: If the trace is empty, out of order, on more than one clock, or has
            no snapshot, or a snapshot names another world or description. Nothing is
            rendered then.
    """
    kinds = [_kind(r) for r in records]
    _check(records, kinds, world)
    t0 = records[0].time_ns

    viewer = MujocoViewer(world.model, port=0, label=label, show_visibility=False)
    viewer.open()
    try:
        server = viewer.server
        assert viewer._scene_mgr is not None
        root = viewer._scene_mgr._body_frames[0]  # the world body: every rendered node is under it
        root.visible = False
        extent = float(world.model.stat.extent)
        anchor = np.asarray(world.model.stat.center, float) + [0.0, 0.0, 0.75 * extent]
        status = _Status()
        text = server.scene.add_label(LABEL, status.text(0), position=anchor, anchor="bottom-left")
        recorder = server.get_scene_serializer()
        data = mujoco.MjData(world.model)

        shown: list[dict[str, Any]] = []
        hidden: dict[str, int] = {}
        frames: list[int] = []
        frame_sequences: list[int] = []
        cursor = t0
        for record, kind in zip(records, kinds, strict=True):
            line = None if kind == "snapshot" else _describe(kind, record.payload)
            if kind != "snapshot" and line is None:
                hidden[kind] = hidden.get(kind, 0) + 1
                continue
            if record.time_ns > cursor:
                recorder.insert_sleep((record.time_ns - cursor) / 1e9)
                cursor = record.time_ns
            if kind == "snapshot":
                world.mapping.apply(record.payload, data)
                viewer.update(data)
                if not frames:
                    root.visible = True
                frames.append(record.time_ns)
                frame_sequences.append(record.sequence)
            else:
                assert line is not None
                status.add(kind, record.payload, line)
                shown.append({"sequence": record.sequence, "time_ns": record.time_ns, "kind": kind, "text": line})
            text.text = status.text(record.time_ns - t0)
        end = records[-1].time_ns
        if end > cursor:
            recorder.insert_sleep((end - cursor) / 1e9)
        recording = recorder.serialize()
    finally:
        viewer.close()

    report = {
        "world": world.world,
        "description": world.description,
        "clock": records[0].clock,
        "trace": {
            "records": len(records),
            "start_ns": t0,
            "end_ns": end,
            "closed": kinds[-1] == "closed",
            "counts": {k: kinds.count(k) for k in sorted(set(kinds))},
        },
        "frames": {
            "source": "snapshot",
            "interpolation": "none",
            "count": len(frames),
            "sequences": frame_sequences,
            "times_ns": frames,
        },
        "intervals_without_frame": _gaps(t0, end, frames),
        "shown": shown,
        "not_shown": dict(sorted(hidden.items())),
        "recording": {
            "duration_s": (end - t0) / 1e9,
            "sha256": hashlib.sha256(recording).hexdigest(),
            "label": LABEL,
        },
        "mj_viser": version("mj-viser"),
    }
    return Replay(recording=recording, report=report)


def _kind(record: TraceRecord) -> str:
    return str(getattr(record.kind, "value", record.kind))


def _check(records: Sequence[TraceRecord], kinds: list[str], world: World) -> None:
    if not records:
        raise ValueError("the trace is empty")
    clock = records[0].clock
    for i, (before, after) in enumerate(zip(records, records[1:], strict=False)):
        if after.sequence <= before.sequence or after.time_ns < before.time_ns:
            raise ValueError(f"trace record {i + 1} is out of order")
    if any(r.clock != clock for r in records):
        raise ValueError("the trace uses more than one clock")
    snapshots = [r.payload for r, k in zip(records, kinds, strict=True) if k == "snapshot"]
    if not snapshots:
        raise ValueError("the trace has no snapshot, so there is no state to show")
    for s in snapshots:
        if s.model != world.world or s.description != world.description:
            raise ValueError(f"a snapshot is of world {s.model!r}, not {world.world!r}")


def _gaps(start: int, end: int, frames: list[int]) -> list[dict[str, int | str]]:
    """Every stretch of trace time that no frame shows: before the first, between two,
    and after the last (until the trace ends)."""
    gaps: list[dict[str, int | str]] = [{"from_ns": start, "to_ns": frames[0], "shows": "nothing"}]
    gaps += [{"from_ns": a, "to_ns": b, "shows": "frame at from_ns"} for a, b in zip(frames, frames[1:], strict=False)]
    gaps.append({"from_ns": frames[-1], "to_ns": end, "shows": "frame at from_ns"})
    return [g for g in gaps if g["to_ns"] != g["from_ns"]]


def _values(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{getattr(value, 'dtype', 'array')} {'×'.join(str(n) for n in shape)}"
    shown = ", ".join(f"{float(v):.3f}" for v in value[:8])
    return f"[{shown}{', …' if len(value) > 8 else ''}]"


def _command(command: Any) -> str:
    inner = getattr(command, "trajectory", None) or command
    target = getattr(inner, "group", None) or getattr(inner, "gripper", None)
    return type(command).__name__ + (f" on {target}" if target else "")


def _describe(kind: str, p: Any) -> str | None:
    """One line for a record, or None for records not shown."""
    if kind == "opened":
        return f"opened {p.runtime} ({p.clock_mode} clock)"
    if kind == "observed":
        return "; ".join(f"{r.channel} = {_values(r.value)}" for r in p.readings)
    if kind == "compiled":
        return f"compiled a trajectory for {p.command.group}"
    if kind == "submitted":
        return f"{p.execution}: submitted {_command(p.command)}"
    if kind == "status":
        why = f" ({p.diagnostic.code})" if p.diagnostic is not None else ""
        return f"{p.execution}: {_enum(p.state)}{why}"
    if kind == "applied":
        if not p.modifications:
            return None
        changes = ", ".join(f"{_enum(m.kind)} {m.target}" for m in p.modifications)
        return f"{p.execution}: applied, {changes}"
    if kind == "health":
        why = f" ({p.diagnostic.code})" if p.diagnostic is not None else ""
        return f"runtime {_enum(p.state)}{why}"
    if kind == "scene":
        held = ", ".join(f"{a.object} held by {a.end_effector}" for a in p.attachments if a.held)
        return f"scene revision {p.revision}" + (f": {held}" if held else "")
    if kind == "violation":
        return f"{p.object} slipped {p.position_error * 1000:.1f} mm, {p.rotation_error:.3f} rad"
    if kind == "closed":
        return "closed"
    return None


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value))


class _Status:
    """What the label shows: time, the scene, each execution's latest event, each
    channel's latest reading, and the latest runtime event."""

    def __init__(self) -> None:
        self._scene = "scene revision 0"
        self._executions: dict[str, dict[str, str]] = {}
        self._channels: dict[str, str] = {}
        self._runtime = ""

    def add(self, kind: str, payload: Any, line: str) -> None:
        if kind == "scene":
            self._scene = line
        elif kind == "observed":
            for r in payload.readings:
                self._channels[r.channel] = f"{r.channel} = {_values(r.value)}"
        elif kind == "submitted":
            self._executions[payload.execution] = {"command": _command(payload.command), "state": "submitted"}
        elif kind == "status":
            why = f" ({payload.diagnostic.code})" if payload.diagnostic is not None else ""
            e = self._executions.setdefault(payload.execution, {"command": "?", "state": "?"})
            e["state"] += f" → {_enum(payload.state)}{why}"
        elif kind == "applied":
            self._executions.setdefault(payload.execution, {"command": "?", "state": "?"})["applied"] = line
        else:
            self._runtime = line

    def text(self, since_start_ns: int) -> str:
        lines = [f"t = {since_start_ns / 1e9:.3f} s", self._scene]
        for execution, e in self._executions.items():
            applied = f"; last {e['applied'].split(': ', 1)[1]}" if "applied" in e else ""
            lines.append(f"{execution}: {e['command']} · {e['state']}{applied}")
        lines += [self._channels[c] for c in sorted(self._channels)]
        if self._runtime:
            lines.append(self._runtime)
        return "\n".join(lines)


def convert_bundle(bundle: str | os.PathLike[str], out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Convert an ssrobot run bundle into ``out_dir``, which must not exist.

    The bundle is verified and its world built by ssrobot before anything renders. The
    files are written beside ``out_dir`` and moved into place only when complete.

    Raises:
        FileExistsError: If ``out_dir`` exists.
        ssrobot.ValidationError: If the bundle does not verify, or its world is not the
            one it recorded.
    """
    target = Path(out_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists")
    ssrobot = importlib.import_module("ssrobot")
    ssrobot_mujoco = importlib.import_module("ssrobot.mujoco")
    opened = ssrobot.open_bundle(bundle)
    world = ssrobot_mujoco.materialize_world(opened)
    replay = record_trace(opened.records, world, label=opened.package.description.name)
    report = replay.report

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        (staging / "recording.viser").write_bytes(replay.recording)
        (staging / "conversion.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mj_viser.replay",
        description="Convert an ssrobot run bundle into a Viser recording (recording.viser, conversion.json).",
    )
    parser.add_argument("bundle", help="run bundle directory")
    parser.add_argument("out_dir", help="output directory; must not exist")
    args = parser.parse_args(argv)
    try:
        report = convert_bundle(args.bundle, args.out_dir)
    except ModuleNotFoundError as e:
        print(f"error: converting a bundle needs ssrobot with its MuJoCo extra ({e})", file=sys.stderr)
        return 2
    except (FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        code = getattr(e, "code", None)
        if code is None:
            raise
        print(f"error: {code}: {e}", file=sys.stderr)
        return 2
    frames = report["frames"]["count"]
    print(f"{args.out_dir}: {frames} frames over {report['recording']['duration_s']:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
