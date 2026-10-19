# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ssrobot planning results as ghost previews.

:class:`PlanPreview` is a :class:`mj_viser.MotionPreview` of ssrobot ``PlanResult`` s in
the scene they were planned in. A result's ghost is its planning snapshot, applied
through the materialized world's snapshot mapping, with the result's joints at the shown
sample and held objects riding their end effectors' declared frames. A supplied invalid
``MotionValidation`` is shown as its first invalid state. Results that do not belong to
the snapshot, world, description, or (live) scene revision are refused, and shown ones
go stale when the scene changes. :func:`record_preview` writes one as a recording.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np

from mj_viser.live import LiveView
from mj_viser.preview import Highlight, JointMotion, MotionPreview, record_motion
from mj_viser.viewer import MujocoViewer

__all__ = ["PlanPreview", "record_preview"]


class PlanPreview(MotionPreview):
    """Ghost previews of ssrobot planning results, chosen and scrubbed per browser.

    Add it to the viewer it previews over: ``live.viewer`` with ``live=``, otherwise a
    viewer of the world's own model.
    """

    def __init__(self, world: Any, description: Any, *, live: LiveView | None = None) -> None:
        """
        Args:
            world: The ssrobot ``MaterializedWorld`` the results were planned in.
            description: The robot description, whose end effectors name the frames
                held objects ride.
            live: The live view it is drawn over, whose world it must be and whose
                scene revision results must match, if any.

        Raises:
            ValueError: If the description or the live view is of another robot or world.
        """
        if description.fingerprint() != world.description:
            raise ValueError("the description is not the one the world was built for")
        if live is not None and live.world != world.world:
            raise ValueError(f"the live view shows world {live.world!r}, not {world.world!r}")
        super().__init__(world.model)
        self._world = world
        self._live = live
        self._end_effectors: dict[str, str] = {str(e.name): str(e.frame) for e in description.end_effectors}
        self._qpos = {name: address for name, address in world.mapping.joints}
        self._objects = dict(world.mapping.objects)
        self._revisions: dict[str, int] = {}

    def name(self) -> str:
        return "Plan preview"

    def add_result(self, name: str, result: Any, snapshot: Any, *, validation: Any = None) -> None:
        """Offer a planning result for preview.

        Args:
            name: How browsers choose it.
            result: A successful ``PlanResult`` with a path or trajectory.
            snapshot: The ``SceneSnapshot`` it was planned in.
            validation: Optionally, an invalid ``MotionValidation`` of this plan's own
                motion, whose first invalid state is shown as supplied.

        Raises:
            ValueError: If the result has nothing to show, or was planned in another
                snapshot, world, description, or (live) scene revision; or the
                validation is valid or of another snapshot, motion, group, or joints; or
                a held object's end effector has no frame in the world.
        """
        motion = self._motion(name, result, snapshot, validation)
        stale = self._stale_reason(result.request.revision)
        if stale is not None:
            raise ValueError(f"{name}: {stale}")
        self._revisions[name] = result.request.revision
        self.add(name, motion)

    def motion_of(self, name: str, result: Any, snapshot: Any, *, validation: Any = None) -> JointMotion:
        """The checked :class:`~mj_viser.JointMotion` of a result, without offering it."""
        return self._motion(name, result, snapshot, validation)

    # -- MotionPreview -----------------------------------------------------------------

    def _accept_viewer(self, viewer: MujocoViewer) -> None:
        if self._live is not None:
            if viewer is not self._live.viewer:
                raise ValueError("add the preview to its live view's viewer")
            return
        super()._accept_viewer(viewer)

    def on_sync(self, viewer: MujocoViewer) -> None:
        for name, revision in list(self._revisions.items()):
            stale = self._stale_reason(revision)
            if stale is not None:
                self.set_stale(name, stale)

    # -- internal ------------------------------------------------------------------------

    def _stale_reason(self, revision: int) -> str | None:
        frame = None if self._live is None else self._live.frame
        if frame is not None and frame.revision != revision:
            return f"planned at scene revision {revision}; the scene is at {frame.revision}"
        return None

    def _motion(self, name: str, result: Any, snapshot: Any, validation: Any) -> JointMotion:
        request = result.request
        if result.path is None and result.trajectory is None:
            status = getattr(result, "status", None)
            why = f"the plan {getattr(status, 'value', status)}" if status is not None else "the result"
            raise ValueError(f"{name}: {why} has no path or trajectory to show")
        if request.snapshot != snapshot.fingerprint():
            raise ValueError(f"{name}: the result was planned in another snapshot")
        if snapshot.model != self._world.world or snapshot.description != self._world.description:
            raise ValueError(f"{name}: the snapshot is of another world or description")
        if request.description != self._world.description:
            raise ValueError(f"{name}: the result is for another robot description")
        unknown = [j for j in request.joints if j not in self._qpos]
        if unknown:
            raise ValueError(f"{name}: the world has no joints {unknown}")
        highlight = None
        if validation is not None:
            if validation.valid or validation.witness is None:
                raise ValueError(f"{name}: only an invalid validation has a state to show")
            validator = validation.validator
            if validator.snapshot != request.snapshot:
                raise ValueError(f"{name}: the validation is bound to another snapshot")
            if validator.group != request.group or tuple(validator.joints) != tuple(request.joints):
                raise ValueError(f"{name}: the validation is of another group or joints")
            if validation.motion not in {v.motion for v in result.validation}:
                raise ValueError(f"{name}: the validation is of another motion than this plan's")
            highlight = _witness(validation.witness)
        held = [a for a in snapshot.attachments if a.held and a.object in self._objects]
        frames = {}
        for attachment in held:
            frame = self._held_frame(attachment)
            if frame is None:
                raise ValueError(
                    f"{name}: no frame for {attachment.object!r}'s end effector {attachment.end_effector!r}"
                )
            frames[attachment.object] = frame

        model, mapping, qpos, objects, joints = (
            self._model,
            self._world.mapping,
            self._qpos,
            self._objects,
            request.joints,
        )

        def state(configuration: Sequence[float], data: mujoco.MjData) -> None:
            # The planning snapshot, the result's joints at the sample, and held objects
            # riding their end effectors, as the planning scene moves them.
            mapping.apply(snapshot, data)
            for joint, value in zip(joints, configuration, strict=True):
                data.qpos[qpos[joint]] = value
            mujoco.mj_kinematics(model, data)
            for attachment in held:
                pose = _frame_pose(model, data, frames[attachment.object]) @ _transform(attachment.transform)
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, pose[:3, :3].flatten())
                at = objects[attachment.object]
                data.qpos[at : at + 3] = pose[:3, 3]
                data.qpos[at + 3 : at + 7] = quat

        trajectory = result.trajectory
        return JointMotion(
            joints=tuple(joints),
            state=state,
            waypoints=tuple(tuple(q) for q in result.path.waypoints) if result.path is not None else None,
            duration_s=trajectory.duration if trajectory is not None else None,
            at=(lambda t: trajectory.evaluate(t)[0]) if trajectory is not None else None,
            samples=(lambda period: list(_samples(trajectory, period))) if trajectory is not None else None,
            carried=tuple(a.object for a in held),
            highlight=highlight,
            notes={
                "path": "the planner's own waypoint",
                "trajectory": "evaluated exactly from the planner's timing law",
            },
        )

    def _held_frame(self, attachment: Any) -> str | None:
        """The MJCF body or site a held object rides: its end effector's frame."""
        frame = self._end_effectors.get(attachment.end_effector)
        if frame is None:
            return None
        for kind in (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE):
            if mujoco.mj_name2id(self._model, kind, frame) >= 0:
                return frame
        return None


def record_preview(
    result: Any,
    snapshot: Any,
    world: Any,
    description: Any,
    *,
    period_ns: int = 20_000_000,
    waypoint_s: float = 0.5,
) -> tuple[bytes, dict[str, Any]]:
    """Record a planning result's preview: :func:`mj_viser.record_motion` of its checked
    motion, whose report also names the snapshot and world.

    Raises:
        ValueError: As :meth:`PlanPreview.add_result`.
    """
    motion = PlanPreview(world, description).motion_of("result", result, snapshot)
    recording, report = record_motion(world.model, motion, period_ns=period_ns, waypoint_s=waypoint_s)
    return recording, {"snapshot": snapshot.fingerprint(), "world": world.world, **report}


def _samples(trajectory: Any, period_ns: int) -> Sequence[tuple[int, Sequence[float]]]:
    command = trajectory.sample(period_ns)
    return list(zip(command.time_from_start_ns, command.positions, strict=True))


def _witness(witness: Any) -> Highlight:
    where = (
        f"at t = {witness.time_s:.3f} s"
        if witness.time_s is not None
        else f"in segment {witness.segment} at {witness.fraction:.3f}"
        if witness.segment is not None
        else "at the state"
    )
    contacts = ", ".join(f"{c.first}/{c.second} {c.distance * 1000:.2f} mm" for c in witness.contacts)
    lines = [f"first invalid state: {_enum(witness.failure)} {where}", "as the validator reported it"]
    if contacts:
        lines.append(f"contacts: {contacts}")
    if witness.detail:
        lines.append(witness.detail)
    return Highlight(configuration=tuple(witness.configuration), lines=tuple(lines), label="first invalid state")


def _frame_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    pose = np.eye(4)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body >= 0:
        pose[:3, :3] = data.xmat[body].reshape(3, 3)
        pose[:3, 3] = data.xpos[body]
        return pose
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    pose[:3, :3] = data.site_xmat[site].reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site]
    return pose


def _transform(pose: Any) -> np.ndarray:
    out = np.eye(4)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(pose.quat_wxyz, float))
    out[:3, :3] = rotation.reshape(3, 3)
    out[:3, 3] = pose.position
    return out


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value))
