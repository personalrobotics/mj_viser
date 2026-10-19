# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The ssrobot integration's trace channels, on stand-in records (needs ssrobot installed).

A stand-in description declares joint positions of a group, joint velocities the runtime
does not provide, a pose, and a depth image. A stand-in run opens, observes, submits,
fails with a diagnostic, and changes its scene. The checks: each channel is displayed,
unavailable, or unsupported as declared; readings are named by joint and component; an
execution shows its whole state history; the scene shows what is held; of many
executions, only the latest three are shown after a count; and a reading goes stale
when trace time passes without another.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

from mj_viser.integrations.ssrobot import TraceChannels

DESCRIPTION = NS(
    groups=(NS(name="arm", joints=("shoulder", "elbow")),),
    channels=(
        NS(name="arm_q", quantity="joint_position", source="arm"),
        NS(name="arm_qd", quantity="joint_velocity", source="arm"),
        NS(name="hand_pose", quantity="pose", source="hand"),
        NS(name="wrist_depth", quantity="depth_image", source="wrist_camera"),
    ),
)


def _reading(channel: str, t: int, value: Any) -> NS:
    return NS(channel=channel, stamp=NS(time_ns=t), value=value)


def _records() -> list[NS]:
    pose = (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0)
    rows = [
        ("opened", 0, NS(runtime="stand-in", clock_mode="manual", channels=("arm_q", "hand_pose", "wrist_depth"))),
        ("observed", 0, NS(readings=(_reading("arm_q", 0, (0.3, 0.1)),))),
        ("submitted", 100_000_000, NS(execution="e1", command=NS(group="arm"))),
        ("status", 100_000_000, NS(execution="e1", state="active", diagnostic=None)),
        (
            "observed",
            500_000_000,
            NS(readings=(_reading("arm_q", 500_000_000, (-0.7, 0.25)), _reading("hand_pose", 500_000_000, pose))),
        ),
        ("status", 600_000_000, NS(execution="e1", state="failed", diagnostic=NS(code="<b>stalled</b>"))),
        ("scene", 600_000_000, NS(revision=1, attachments=(NS(object="cup", end_effector="hand", held=True),))),
    ]
    return [NS(sequence=i, kind=k, time_ns=t, payload=p) for i, (k, t, p) in enumerate(rows)]


def check_trace_channels() -> dict[str, Any]:
    out: dict[str, Any] = {}
    channels = TraceChannels(DESCRIPTION, stale_after_ns=1_000_000_000)
    for record in _records():
        channels(record)  # the sink interface
    coverage = {name: c["status"] for name, c in channels.board.coverage().items()}
    expected = {"arm_q": "displayed", "arm_qd": "unavailable", "hand_pose": "displayed", "wrist_depth": "unsupported"}
    if coverage != expected:
        raise SystemExit(f"channel coverage {coverage}, expected {expected}")
    out["coverage"] = coverage
    lines = channels.board.lines()
    for expect in (
        "arm_q = shoulder -0.700, elbow 0.250 (t = 0.500 s)",
        "hand_pose = x 0.100, y 0.200, z 0.300, qw 1.000, qx 0.000, qy 0.000, qz 0.000 (t = 0.500 s)",
        "e1: SimpleNamespace on arm · submitted → active → failed (<b>stalled</b>)",  # escaped in markup
        "scene revision 1: cup held by hand",
    ):
        if expect not in lines:
            raise SystemExit(f"{expect!r} not among {lines}")
    # Many executions: only the latest three are shown, so the panel keeps its size.
    for i in range(2, 6):
        channels(
            NS(
                sequence=90 + i,
                kind="submitted",
                time_ns=700_000_000,
                payload=NS(execution=f"e{i}", command=NS(group="<i>arm</i>" if i == 5 else "arm")),
            )
        )
    shown = [line for line in channels.board.lines() if line.startswith(("e", "…"))]
    if shown[0] != "…: 2 earlier" or [x.split(":")[0] for x in shown[1:]] != ["e3", "e4", "e5"]:
        raise SystemExit(f"executions shown: {shown}")
    out["executions_shown"] = shown[0]
    channels(NS(sequence=99, kind="stepped", time_ns=1_600_000_000, payload=None))
    status = {c.name: c.status for c in channels.board.channels()}
    if status["arm_q"] != "stale" or status["arm_qd"] != "unavailable":
        raise SystemExit(f"after 1.1 s without readings: {status}")
    out["stale_after_1s"] = status["arm_q"]
    return out
