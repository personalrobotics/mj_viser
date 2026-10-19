# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Replay an ssrobot run as a native Viser recording.

The run's trace is the record; a ``.viser`` file made from it is a derived view. Frames
come from the trace's ``SceneSnapshot`` records, at whatever cadence the run took them,
and nothing is interpolated between them. A :class:`TraceChannels` board of the trace
is shown in the recording's text and images at each record's time. Playback uses
Viser's own player and timeline.

``python -m mj_viser.integrations.ssrobot.replay BUNDLE OUT_DIR`` converts a run
bundle: it opens and verifies it with ``ssrobot.open_bundle``, builds its world with
``ssrobot.mujoco.materialize_world`` (no runtime, context, or physics), and writes
``recording.viser`` and ``conversion.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mujoco

from mj_viser.integrations.ssrobot import materialize_world, open_bundle
from mj_viser.integrations.ssrobot.channels import TraceChannels
from mj_viser.recording import IMAGES, TEXT, StateRecorder

__all__ = ["Replay", "convert_bundle", "record_trace"]


@dataclass(frozen=True)
class Replay:
    """A converted run."""

    recording: bytes
    """The ``.viser`` file."""
    report: dict[str, Any]
    """What was shown when, what was not, and every interval without a frame."""


def record_trace(records: Sequence[Any], world: Any, *, description: Any = None, label: str | None = None) -> Replay:
    """Convert a trace of ``world`` (an ssrobot ``MaterializedWorld``) into a recording.

    Recording time is trace time since the first record. Before the first snapshot the
    model is hidden: there is no recorded state to show.

    Raises:
        ValueError: If the trace is empty, out of order, on more than one clock, or has
            no snapshot, or a snapshot names another world or description. Nothing is
            rendered then.
    """
    kinds = [_kind(r) for r in records]
    _check(records, kinds, world)
    t0, end = records[0].time_ns, records[-1].time_ns
    trace = TraceChannels(description)
    board = trace.board
    data = mujoco.MjData(world.model)
    shown: list[dict[str, Any]] = []
    hidden: dict[str, int] = {}
    hidden_changes: list[dict[str, Any]] = []
    images: dict[str, dict[str, list[int]]] = {}
    frames: list[int] = []
    frame_sequences: list[int] = []
    with StateRecorder(world.model, label=label) as recorder:
        recorder.advance(t0)
        recorder.text(board.lines())
        showing = board.lines()[1:]  # what the text shows, apart from the time
        for record, kind in zip(records, kinds, strict=True):
            trace(record)
            # ``stepped`` marks time only; an applied setpoint is shown only when modified.
            # Either can still change what the text shows (a reading going stale), and
            # only then is it updated.
            if kind == "stepped" or (kind == "applied" and not record.payload.modifications):
                hidden[kind] = hidden.get(kind, 0) + 1
                if board.lines()[1:] == showing:
                    continue
                hidden_changes.append({"sequence": record.sequence, "time_ns": record.time_ns, "kind": kind})
            recorder.advance(record.time_ns)
            if kind == "snapshot":
                world.mapping.apply(record.payload, data)
                recorder.show(data)
                frames.append(record.time_ns)
                frame_sequences.append(record.sequence)
            elif not hidden_changes or hidden_changes[-1]["sequence"] != record.sequence:
                shown.append({"sequence": record.sequence, "time_ns": record.time_ns, "kind": kind})
            if kind == "observed":
                _show_images(recorder, trace, record, images)
            lines = board.lines()
            recorder.text(lines)
            showing = lines[1:]
        recorder.advance(end)
        recording = recorder.finish()

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
        "channels": board.coverage(),
        "not_shown": dict(sorted(hidden.items())),
        "not_shown_but_changed_the_label": hidden_changes,
        "images": {name: images[name] for name in sorted(images)},
        # What a trace cannot show: ssrobot records a contact channel's touching and
        # summed normal force (shown above), not contact points, normals, or forces.
        "contact_geometry": "not in the trace",
        "recording": {
            "duration_s": (end - t0) / 1e9,
            "sha256": hashlib.sha256(recording).hexdigest(),
            "label": TEXT,
            "images": IMAGES,
        },
        "mj_viser": version("mj-viser"),
    }
    return Replay(recording=recording, report=report)


def _show_images(
    recorder: StateRecorder, trace: TraceChannels, record: Any, images: dict[str, dict[str, list[int]]]
) -> None:
    """Show each image reading of an observation; account for every one, shown or
    malformed. A malformed reading hides its channel's image rather than leave an older
    one showing as if it were this reading."""
    image_channels = {c.name for c in trace.board.channels() if c.kind == "image"}
    for reading in record.payload.readings:
        if reading.channel not in image_channels:
            continue
        stamp = reading.stamp.time_ns
        log = images.setdefault(reading.channel, {"readings": [], "shown": [], "malformed": []})
        log["readings"].append(stamp)
        image, _ = trace.board.image(reading.channel)
        log["shown" if image is not None else "malformed"].append(stamp)
        recorder.image(reading.channel, image)


def _kind(record: Any) -> str:
    return str(getattr(record.kind, "value", record.kind))


def _check(records: Sequence[Any], kinds: list[str], world: Any) -> None:
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
    opened = open_bundle(bundle)
    world = materialize_world(opened)
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
        prog="python -m mj_viser.integrations.ssrobot.replay",
        description="Convert an ssrobot run bundle into a Viser recording (recording.viser, conversion.json).",
    )
    parser.add_argument("bundle", help="run bundle directory")
    parser.add_argument("out_dir", help="output directory; must not exist")
    args = parser.parse_args(argv)
    try:
        report = convert_bundle(args.bundle, args.out_dir)
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
