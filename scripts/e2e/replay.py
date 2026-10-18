# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Trace replay's contract with any trace and world, without ssrobot.

A stand-in trace, shaped like ssrobot's, of the fixture model: two snapshots of states A
and B, a submitted command and its statuses between them. The checks, on the recording:
each frame shows MuJoCo's kinematics of its snapshot at the snapshot's time; the label
shows each event at its recorded time; the report accounts for every interval without a
frame; a trace of another world, or with no snapshot, is refused before anything
renders; and the conversion command without ssrobot says what it needs.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace as NS
from typing import Any

import mujoco

from e2e.fixtures import STATES, expect_raises, kinematics
from e2e.oracles import check_world_poses, recorded_local_poses, replay_recording
from mj_viser import MujocoViewer
from mj_viser.replay import LABEL, record_trace


@dataclass(frozen=True)
class _Record:
    sequence: int
    kind: str
    clock: str
    time_ns: int
    payload: Any = None


class _Mapping:
    def apply(self, snapshot: Any, data: mujoco.MjData) -> None:
        data.qpos[:] = snapshot.positions


def _snapshot(positions: list[float], world: str = "w") -> NS:
    return NS(model=world, description="d", positions=positions)


def _trace(world: str = "w") -> list[_Record]:
    command = NS(group="arm")
    rows = [
        ("opened", 0, NS(runtime="stand-in", clock_mode="manual")),
        ("snapshot", 0, _snapshot(STATES["A"], world)),
        ("submitted", 10_000_000, NS(execution="e1", command=command)),
        ("status", 10_000_000, NS(execution="e1", state="active", diagnostic=None)),
        ("stepped", 20_000_000, None),
        ("snapshot", 20_000_000, _snapshot(STATES["B"], world)),
        ("status", 30_000_000, NS(execution="e1", state="succeeded", diagnostic=None)),
        ("closed", 30_000_000, None),
    ]
    return [_Record(i, kind, "sim", t, p) for i, (kind, t, p) in enumerate(rows)]


def check_replay(model: mujoco.MjModel) -> dict[str, Any]:
    world = NS(model=model, world="w", description="d", mapping=_Mapping())
    replay = record_trace(_trace(), world)
    report = replay.report
    if report["frames"]["times_ns"] != [0, 20_000_000]:
        raise SystemExit(f"frames at {report['frames']['times_ns']}")
    if report["intervals_without_frame"] != [
        {"from_ns": 0, "to_ns": 20_000_000, "shows": "frame at from_ns"},
        {"from_ns": 20_000_000, "to_ns": 30_000_000, "shows": "frame at from_ns"},
    ]:
        raise SystemExit(f"intervals without a frame: {report['intervals_without_frame']}")

    messages = replay_recording(replay.recording)
    scene = MujocoViewer(model).scene_report()
    for name, before in (("A", 0.015), ("B", float("inf"))):
        check_world_poses(recorded_local_poses(messages, before), scene, model, kinematics(model, STATES[name]), name)
    texts = [(t, (m.get("props") or m.get("updates") or {}).get("text")) for t, m in messages if m.get("name") == LABEL]
    texts = [(t, s) for t, s in texts if s is not None]

    def at(t: float) -> str:
        return [s for when, s in texts if when <= t + 1e-9][-1]

    for t, expect in ((0.01, "e1: SimpleNamespace on arm · submitted → active"), (0.03, "→ active → succeeded")):
        if expect not in at(t):
            raise SystemExit(f"at {t} s the label reads {at(t)!r}, not {expect!r}")

    out = {
        "frames": report["frames"]["count"],
        "other_world": expect_raises(ValueError, lambda: record_trace(_trace("v"), world)),
        "no_snapshot": expect_raises(
            ValueError, lambda: record_trace([r for r in _trace() if r.kind != "snapshot"], world)
        ),
        "empty": expect_raises(ValueError, lambda: record_trace([], world)),
    }
    try:
        import ssrobot  # noqa: F401
    except ModuleNotFoundError:
        result = subprocess.run(
            [sys.executable, "-I", "-m", "mj_viser.replay", "bundle", "out"], capture_output=True, text=True
        )
        if result.returncode != 2 or "needs ssrobot" not in result.stderr:
            raise SystemExit(f"the conversion command without ssrobot: {result}")
        out["command_without_ssrobot"] = result.stderr.strip()
    return out
