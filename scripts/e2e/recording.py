# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Recording caller-supplied states, without ssrobot.

A StateRecorder of the fixture model: the model is hidden until the first state, states
A and B show at their times, the text changes with them, and time cannot go back. The
recording itself is decoded: each state's frame poses are MuJoCo's kinematics of it.
The ssrobot integration's replay command, run without ssrobot, says what it needs and
writes nothing.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mujoco

from e2e.fixtures import STATES, expect_raises, kinematics
from e2e.oracles import check_world_poses, recorded_local_poses, replay_recording
from mj_viser import MujocoViewer, StateRecorder
from mj_viser.recording import TEXT


def check_recording(model: mujoco.MjModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with StateRecorder(model) as recorder:
        recorder.advance(1_000_000_000)  # the recording starts here
        recorder.text(["nothing yet"])
        recorder.advance(1_100_000_000)
        recorder.show(kinematics(model, STATES["A"]))
        recorder.text(["state A"])
        recorder.advance(1_300_000_000)
        recorder.show(kinematics(model, STATES["B"]))
        recorder.text(["state B"])
        out["time_goes_back"] = expect_raises(ValueError, lambda: recorder.advance(1_200_000_000))
        recording = recorder.finish()
    messages = replay_recording(recording)
    root_hidden = [
        (round(t, 6), m["visible"])
        for t, m in messages
        if m["type"] == "SetSceneNodeVisibilityMessage" and m.get("name", "").endswith("/b0_world")
    ]
    if root_hidden[:1] != [(0.0, False)] or (0.1, True) not in root_hidden:
        raise SystemExit(f"the model's visibility over the recording: {root_hidden}")
    scene = MujocoViewer(model).scene_report()
    for name, before in (("A", 0.2), ("B", float("inf"))):
        check_world_poses(recorded_local_poses(messages, before), scene, model, kinematics(model, STATES[name]), name)
    texts = [
        (round(t, 6), (m.get("props") or m.get("updates") or {}).get("text"))
        for t, m in messages
        if m.get("name") == TEXT
    ]
    texts = [(t, s) for t, s in texts if s]
    if texts != [(0.0, "nothing yet"), (0.1, "state A"), (0.3, "state B")]:
        raise SystemExit(f"recorded text {texts}")
    out["states"] = ["A at 0.1 s", "B at 0.3 s"]

    # The ssrobot replay command without ssrobot: a clear failure, and no output.
    with tempfile.TemporaryDirectory() as root:
        dest = Path(root) / "replay"
        result = subprocess.run(
            [sys.executable, "-I", "-m", "mj_viser.integrations.ssrobot.replay", str(Path(root) / "bundle"), str(dest)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 or "needs ssrobot" not in result.stderr or any(Path(root).iterdir()):
            raise SystemExit(f"the replay command without ssrobot: {result}")
    out["replay_command_without_ssrobot"] = result.stderr.strip().splitlines()[-1]
    return out
