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

from mj_viser.channels import ChannelModel
from mj_viser.viewer import MujocoViewer

LABEL = "/replay/status"
"""Scene path of the label that shows the trace's events."""


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


def record_trace(
    records: Sequence[TraceRecord], world: World, *, description: Any = None, label: str | None = None
) -> Replay:
    """Convert a trace of ``world`` into a Viser recording.

    Recording time is trace time since the first record. Before the first snapshot the
    world is hidden: there is no recorded state to show. The label shows what a
    :class:`~mj_viser.channels.ChannelModel` of the trace shows at each record's time;
    ``description`` (the robot description) gives it channel quantities and joint names.

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
        channels = ChannelModel(description)
        text = server.scene.add_label(LABEL, "\n".join(channels.lines()), position=anchor, anchor="bottom-left")
        recorder = server.get_scene_serializer()
        data = mujoco.MjData(world.model)

        shown: list[dict[str, Any]] = []
        hidden: dict[str, int] = {}
        hidden_changes: list[dict[str, Any]] = []
        showing = channels.lines()[1:]  # what the label shows, apart from the time
        frames: list[int] = []
        frame_sequences: list[int] = []
        cursor = t0
        for record, kind in zip(records, kinds, strict=True):
            channels.accept(record)
            # ``stepped`` marks time only; an applied setpoint is shown only when modified.
            # Either can still change what the label shows (a reading going stale), and
            # only then is the label updated.
            if kind == "stepped" or (kind == "applied" and not record.payload.modifications):
                hidden[kind] = hidden.get(kind, 0) + 1
                if channels.lines()[1:] == showing:
                    continue
                hidden_changes.append({"sequence": record.sequence, "time_ns": record.time_ns, "kind": kind})
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
            elif not hidden_changes or hidden_changes[-1]["sequence"] != record.sequence:
                shown.append({"sequence": record.sequence, "time_ns": record.time_ns, "kind": kind})
            lines = channels.lines()
            text.text = "\n".join(lines)
            showing = lines[1:]
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
        "channels": channels.coverage(),
        "not_shown": dict(sorted(hidden.items())),
        "not_shown_but_changed_the_label": hidden_changes,
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


def convert_bundle(bundle: str | os.PathLike[str], out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Convert an ssrobot run bundle into ``out_dir``, which must not exist.

    The bundle is verified and its world built by ssrobot before anything renders. The
    files are written beside ``out_dir`` and moved into place only when complete.

    Raises:
        FileExistsError: If ``out_dir`` exists.
        ValueError: If ``out_dir`` is inside the bundle, symlinks resolved: writing there
            would leave the bundle with undeclared files.
        ssrobot.ValidationError: If the bundle does not verify, or its world is not the
            one it recorded.
    """
    target = Path(out_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists")
    source = Path(bundle).resolve()
    if target.resolve().is_relative_to(source):
        raise ValueError(f"{target} is inside the bundle {bundle}; write the replay elsewhere")
    ssrobot = importlib.import_module("ssrobot")
    ssrobot_mujoco = importlib.import_module("ssrobot.mujoco")
    opened = ssrobot.open_bundle(bundle)
    world = ssrobot_mujoco.materialize_world(opened)
    description = opened.package.description
    replay = record_trace(opened.records, world, description=description, label=description.name)
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
