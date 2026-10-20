# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""A downstream program, written against mj_viser's public API, for strict type checking.

It is type-checked, not run: scripts/typing/check.sh installs a wheel into a fresh
environment and runs mypy --strict on it, so mj_viser's own annotations (py.typed) are
what the checker sees.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mj_viser import (
    ChannelBoard,
    ChannelState,
    ContactOverlay,
    Highlight,
    JointMotion,
    LiveView,
    MotionPreview,
    MujocoViewer,
    RenderFrame,
    RenderSource,
    SourceStale,
    StateRecorder,
    contact_set,
)
from mj_viser.contacts import Contact, ContactSet


class Frame:
    def __init__(self, time_ns: int, revision: int, world: str) -> None:
        self.time_ns = time_ns
        self.revision = revision
        self.world = world


class Source:
    """A render source of the caller's own: structurally a RenderSource."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.world = "pendulum"

    def capture(self) -> RenderFrame:
        if self.data.time < 0:
            raise SourceStale("time went back")
        return Frame(int(self.data.time * 1e9), 0, self.world)


def view(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    viewer = MujocoViewer(model, port=0)
    url: str = viewer.open()
    viewer.add_panel(ContactOverlay(scale_m_per_n=0.02))
    viewer.update(data)
    viewer.close()
    return url


def follow(source: RenderSource) -> int | None:
    board = ChannelBoard()
    board.declare("arm_q", quantity="joint_position", labels=("shoulder", "elbow"))
    board.reading("arm_q", 0, (0.3, 0.1))
    board.set_status("task", "reaching")
    states: list[ChannelState] = board.channels()
    live = LiveView(source, channels=board, max_rate_hz=30)
    live.open()
    frame = live.refresh()
    live.close()
    return None if frame is None else frame.time_ns + len(states)


def strongest(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    contacts: ContactSet = contact_set(model, data, limit=10)
    forces = [float(np.linalg.norm(c.force)) for c in contacts.contacts if c.force is not None]
    first: Contact | None = contacts.contacts[0] if contacts.contacts else None
    return max(forces, default=0.0) + (0.0 if first is None else first.distance)


def preview(model: mujoco.MjModel) -> MotionPreview:
    def state(q: object, data: mujoco.MjData) -> None:
        data.qpos[:] = q

    motion = JointMotion(
        joints=("hinge",),
        state=state,
        waypoints=((0.0,), (1.0,)),
        highlight=Highlight(configuration=(1.0,), label="limit", lines=("at its limit",)),
    )
    previews = MotionPreview(model)
    previews.add("swing", motion)
    return previews


def record(model: mujoco.MjModel, data: mujoco.MjData) -> bytes:
    with StateRecorder(model, contacts=True) as recorder:
        recorder.advance(0)
        recorder.show(data)
        recorder.text(["t = 0 s"])
        return recorder.finish()


if __name__ == "__main__":
    floor = "<mujoco><worldbody><geom size='1 1 .1' type='plane'/></worldbody></mujoco>"
    pendulum = mujoco.MjModel.from_xml_string(floor)
    follow(Source(pendulum))
