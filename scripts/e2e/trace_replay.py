# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The ssrobot integration's trace replay, on stand-in records (needs ssrobot installed).

A stand-in trace, shaped like ssrobot's, of the fixture model: two snapshots of states A
and B, a submitted command and its statuses between them. The checks, on the recording:
each frame shows MuJoCo's kinematics of its snapshot at the snapshot's time; the label
shows each event at its recorded time; the report accounts for every interval without a
frame; and a trace of another world, or with no snapshot, is refused before anything
renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace as NS
from typing import Any

import mujoco

from e2e.fixtures import STATES, expect_raises, kinematics
from e2e.oracles import check_world_poses, recorded_local_poses, replay_recording
from mj_viser import MujocoViewer
from mj_viser.integrations.ssrobot.replay import record_trace
from mj_viser.recording import TEXT as LABEL


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
        ("opened", 0, NS(runtime="stand-in", clock_mode="manual", channels=())),
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

    # An interrupted run: its last change is a reading going stale, by elapsed time alone.
    # The label must end saying so, as the report's model does, and must not be updated
    # for steps that change nothing it shows.
    reading = NS(channel="arm_q", stamp=NS(time_ns=0), value=(0.3, 0.1))
    rows = [
        ("opened", 0, NS(runtime="stand-in", clock_mode="manual", channels=("arm_q",))),
        ("snapshot", 0, _snapshot(STATES["A"])),
        ("observed", 0, NS(readings=(reading,))),
        *(("stepped", t, None) for t in (500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000)),
    ]
    interrupted = record_trace([_Record(i, k, "sim", t, p) for i, (k, t, p) in enumerate(rows)], world)
    texts = [
        (t, (m.get("props") or m.get("updates") or {}).get("text"))
        for t, m in replay_recording(interrupted.recording)
        if m.get("name") == LABEL
    ]
    final = [s for _, s in texts if s is not None][-1]
    if "arm_q = [0.300, 0.100] (t = 0.000 s, stale)" not in final.split("\n"):
        raise SystemExit(f"an interrupted trace's label ends {final!r}")
    changes = interrupted.report["not_shown_but_changed_the_label"]
    if [c["time_ns"] for c in changes] != [1_500_000_000]:
        raise SystemExit(f"label updates from hidden records: {changes}")
    out["stale_by_elapsed_time"] = {
        "label": "arm_q = [0.300, 0.100] (t = 0.000 s, stale)",
        "updated_at_ns": 1_500_000_000,
    }
    return out
