# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""LiveView's contract with any render source, without a robot runtime.

A stand-in render source has its own model and data copies,
a scene identity, and capture() that copies the "simulation" state in. The checks:
frames are rendered and coalesced to max_rate_hz; refresh never raises once the view is
closed; SourceStale from the source stops the view, while any other error (even one
carrying another library's stale code) propagates; a scene
change stops it; and bad arguments are refused before anything renders.
"""

from __future__ import annotations

import copy
import socket
import time
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

import mj_viser
from e2e.fixtures import expect_raises, free_port


@dataclass(frozen=True)
class _Frame:
    time_ns: int
    revision: int
    world: str


class _SourceError(Exception):
    """An error with a code, like ssrobot's: not mj_viser's stale signal."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _Source:
    """A render source over a simulation it owns, like ssrobot's RenderHandle."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self._sim_model = model
        self.sim = mujoco.MjData(model)
        self.model = copy.deepcopy(model)
        self.data = mujoco.MjData(self.model)
        self.world = "scene-a"
        self.fail_with: Exception | None = None
        self.next_world: str | None = None

    def capture(self) -> _Frame:
        if self.fail_with is not None:
            raise self.fail_with
        mujoco.mj_copyData(self.data, self._sim_model, self.sim)
        return _Frame(int(self.sim.time * 1e9), 0, self.next_world or self.world)


def check_live_view(model: mujoco.MjModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    source = _Source(model)
    out["other_scene"] = expect_raises(ValueError, lambda: mj_viser.LiveView(source, expected_world="scene-b"))
    out["bad_rate"] = expect_raises(ValueError, lambda: mj_viser.LiveView(source, max_rate_hz=0))

    view = mj_viser.LiveView(source, max_rate_hz=1.0, port=0)
    if view.refresh() is not None:
        raise SystemExit("a closed view rendered")
    view.open()  # renders the current state once
    source.sim.qpos[0] = 0.4
    if view.refresh() is not None:
        raise SystemExit("a refresh within 1/max_rate_hz of the last frame was not coalesced")
    if view.refresh(force=True) is None or not np.isclose(view.viewer.data.qpos[0], 0.4):
        raise SystemExit("a forced refresh did not render the simulation's state")
    time.sleep(1.05)
    source.sim.qpos[0] = 0.8
    if view.refresh() is None or not np.isclose(view.viewer.data.qpos[0], 0.8):
        raise SystemExit("a refresh after 1/max_rate_hz did not render")
    out["coalesced_to_max_rate"] = True

    source.fail_with = RuntimeError("a programming error in the source")
    out["other_errors_propagate"] = expect_raises(RuntimeError, lambda: view.refresh(force=True))
    # An error that merely carries a code, as another library's would, is not a stale
    # signal: only SourceStale is.
    source.fail_with = _SourceError("stale_render")
    out["coded_error_propagates"] = expect_raises(_SourceError, lambda: view.refresh(force=True))
    source.fail_with = mj_viser.SourceStale("stale_render")
    if view.refresh(force=True) is not None or view.stale != "stale_render":
        raise SystemExit(f"a stale source left the view {view.stale!r}")
    if view.refresh(force=True) is not None:
        raise SystemExit("a stale view rendered again")
    view.close()
    if view.refresh(force=True) is not None:
        raise SystemExit("refresh after close rendered")
    out["stale_source"] = view.stale

    # A failed attach is a failed lifecycle transition: the viewer closes and frees its port.
    failing = _Source(model)
    failing.fail_with = RuntimeError("capture failed")
    port = free_port()
    view = mj_viser.LiveView(failing, max_rate_hz=None, port=port)
    out["open_with_failing_source"] = expect_raises(RuntimeError, view.open)
    if view.is_open:
        raise SystemExit("open() left the viewer open after the first frame failed")
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))  # released
    out["failed_open_released_port"] = True

    # A first frame that cannot be shown is a failed open too, not the model's default
    # state behind a URL: a stale source, or a frame from another world.
    stale_first = _Source(model)
    stale_first.fail_with = mj_viser.SourceStale("stale_render")
    other_first = _Source(model)
    other_first.next_world = "scene-b"
    for case, source in (("stale_at_open", stale_first), ("other_world_at_open", other_first)):
        port = free_port()
        view = mj_viser.LiveView(source, max_rate_hz=None, port=port)
        out[case] = expect_raises(mj_viser.SourceStale, view.open)
        if view.is_open:
            raise SystemExit(f"{case}: open() returned with the viewer open")
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))  # released

    changed = _Source(model)
    view = mj_viser.LiveView(changed, max_rate_hz=None, port=0)
    view.open()
    changed.next_world = "scene-b"
    if view.refresh() is not None or view.stale != "world_changed":
        raise SystemExit(f"a frame from another scene left the view {view.stale!r}")
    view.close()
    out["scene_change"] = "world_changed"
    return out
